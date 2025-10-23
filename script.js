// Import necessary components from the MediaPipe Vision Tasks library
        import {
            FilesetResolver,
            FaceDetector
        } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";

        // --- Constants ---
        // Max distance in pixels for a box center to be considered the same face across frames
        const PROXIMITY_THRESHOLD = 100; 
        const CENTER_LINE_X_THRESHOLD = 0.5; // Normalized X coordinate (center of canvas)
        const MAX_MISS_FRAMES = 5; // Number of frames to persist a track without detection

        // --- DOM Elements ---
        const webcamElement = document.getElementById("webcam");
        const canvasElement = document.getElementById("canvas");
        const canvasCtx = canvasElement.getContext("2d");
        const statusMessage = document.getElementById("status-message");
        const statusText = document.getElementById("status-text");
        const detectionCount = document.getElementById("detection-count");
        // Removed videoContainer query here, using the newly styled wrapper
        const enterCountElement = document.getElementById("enter-count");
        const exitCountElement = document.getElementById("exit-count");

        // --- State Variables ---
        let faceDetector = null;
        let lastVideoTime = -1;
        let enterCount = 0; // L to R crossing
        let exitCount = 0;  // R to L crossing
        
        /** * Tracks face state across frames. 
         * Key: unique ID (number). 
         * Value: { 
         * prevCenterFlippedX: number, 
         * lastSide: 'left' | 'right', 
         * isEnterCounted: boolean, 
         * isExitCounted: boolean,
         * missedFrames: number, // Counter for consecutive frames where detection was missed
         * boundingBox: object // Stores the last known box data for drawing and calculation
         * }
         */
        let trackedFaceStates = new Map();
        let nextTrackingId = 0;

        // Utility function for updating UI status
        function updateStatus(text, isError = false) {
            statusText.textContent = text;
            statusMessage.style.opacity = '1';
            statusMessage.classList.toggle('bg-red-700', isError);
            statusMessage.classList.toggle('bg-gray-900', !isError);
            document.getElementById("loading-spinner").style.display = isError ? 'none' : 'block';
        }

        function hideStatus() {
            statusMessage.style.opacity = '0';
        }
        
        // Helper to calculate the visual center of the face box on the screen (after flipping)
        function getFlippedBoxCenter(boundingBox, canvasWidth) {
            // The visual center on the screen
            const flippedX = canvasWidth - boundingBox.originX - boundingBox.width;
            return flippedX + boundingBox.width / 2;
        }


        // 1. Initialize the FaceDetector model
        async function createFaceDetector() {
            try {
                updateStatus("Loading required libraries...");
                
                const vision = await FilesetResolver.forVisionTasks(
                    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
                );

                updateStatus("Loading face detection model...");

                faceDetector = await FaceDetector.createFromOptions(vision, {
                    baseOptions: {
                        modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite`,
                        delegate: "GPU" // Use GPU for better performance if available
                    },
                    minDetectionConfidence: 0.5, 
                    runningMode: "VIDEO"
                });

                updateStatus("Model loaded. Starting webcam...");
                await startWebcam();

            } catch (error) {
                console.error("Failed to initialize FaceDetector:", error);
                updateStatus(`Initialization Error: ${error.message}. Check console for details.`, true);
            }
        }

        // 2. Start the user's webcam
        async function startWebcam() {
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                updateStatus("Browser does not support camera access.", true);
                return;
            }

            try {
                const stream = await navigator.mediaDevices.getUserMedia({ video: true });
                webcamElement.srcObject = stream;
                
                // Once metadata is loaded, set canvas size and start detection loop
                webcamElement.onloadedmetadata = () => {
                    // Set video container size based on video stream dimensions
                    const { videoWidth, videoHeight } = webcamElement;
                    
                    // Find the video container (parent) to set its height based on width for correct aspect ratio
                    const videoContainer = webcamElement.closest('.video-container');
                    if (videoContainer) {
                        // Use padding-bottom trick for intrinsic aspect ratio control
                        videoContainer.style.paddingBottom = `${(videoHeight / videoWidth) * 100}%`;
                    }
                    
                    // Set canvas size to match video size
                    canvasElement.width = videoWidth;
                    canvasElement.height = videoHeight;
                    
                    hideStatus();
                    window.requestAnimationFrame(detectFaces);
                };

            } catch (error) {
                console.error("Error accessing webcam:", error);
                updateStatus("Webcam Error: Please ensure camera permissions are granted.", true);
            }
        }

        // 3. Main detection loop
        function detectFaces() {
            if (!faceDetector || !webcamElement.videoWidth) {
                window.requestAnimationFrame(detectFaces);
                return;
            }
            
            // --- 1. Run Detection ---
            let detectionResult = { detections: [] };
            if (webcamElement.currentTime !== lastVideoTime) {
                detectionResult = faceDetector.detectForVideo(webcamElement, performance.now());
                lastVideoTime = webcamElement.currentTime;
            }
            
            const centerLineX = canvasElement.width * CENTER_LINE_X_THRESHOLD;
            const newDetections = detectionResult.detections;
            
            // Prepare for the next frame's tracking data
            const NEXT_FRAME_TRACKED_FACES = new Map();
            const unmatchedDetections = [...newDetections];

            // --- Drawing Setup (Clear and Draw Video Frame) ---
            canvasCtx.save();
            canvasCtx.scale(-1, 1); 
            canvasCtx.translate(-canvasElement.width, 0);
            canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
            canvasCtx.drawImage(webcamElement, 0, 0, canvasElement.width, canvasElement.height);
            canvasCtx.restore();

            // --- Draw Center Line ---
            canvasCtx.save();
            canvasCtx.beginPath();
            canvasCtx.strokeStyle = 'yellow';
            canvasCtx.lineWidth = 3;
            canvasCtx.setLineDash([10, 5]); 
            canvasCtx.moveTo(centerLineX, 0);
            canvasCtx.lineTo(centerLineX, canvasElement.height);
            canvasCtx.stroke();
            canvasCtx.restore();
            
            
            // --- 2. PERSISTENCE & MATCHING (Iterate over OLD tracks) ---
            for (const [id, oldState] of trackedFaceStates.entries()) {
                let matchFound = false;
                let bestMatchIndex = -1;
                let minDistance = Infinity;

                // Find the best matching new detection based on proximity
                for (let i = 0; i < unmatchedDetections.length; i++) {
                    const detection = unmatchedDetections[i];
                    const currentCenterFlippedX = getFlippedBoxCenter(detection.boundingBox, canvasElement.width);
                    const distance = Math.abs(currentCenterFlippedX - oldState.prevCenterFlippedX);
                    
                    if (distance < PROXIMITY_THRESHOLD && distance < minDistance) {
                        minDistance = distance;
                        bestMatchIndex = i;
                        matchFound = true;
                    }
                }
                
                // --- A. Match Found: Update State ---
                if (matchFound) {
                    const matchingDetection = unmatchedDetections.splice(bestMatchIndex, 1)[0];
                    const newBoundingBox = matchingDetection.boundingBox;
                    const currentCenterFlippedX = getFlippedBoxCenter(newBoundingBox, canvasElement.width);
                    
                    // Create the next state object
                    const currentState = {
                        ...oldState,
                        prevCenterFlippedX: currentCenterFlippedX,
                        missedFrames: 0, // Reset miss counter
                        boundingBox: newBoundingBox, // Store the newly detected box
                    };
                    
                    // Run COUNTING LOGIC (Immediate Count based on Center Crossing)
                    const currentSide = currentCenterFlippedX < centerLineX ? 'left' : 'right';
                    
                    if (currentSide !== currentState.lastSide) {
                        // Crossing detected!
                        if (currentState.lastSide === 'left' && currentSide === 'right' && !currentState.isEnterCounted) {
                            // L -> R Crossing
                            enterCount++;
                            currentState.isEnterCounted = true;
                        } else if (currentState.lastSide === 'right' && currentSide === 'left' && !currentState.isExitCounted) {
                            // R -> L Crossing
                            exitCount++;
                            currentState.isExitCounted = true;
                        }
                        
                        // Update lastSide only after the counting checks
                        currentState.lastSide = currentSide;
                    }

                    NEXT_FRAME_TRACKED_FACES.set(id, currentState);
                    
                // --- B. Match NOT Found: Handle Persistence ---
                } else {
                    const newMissedFrames = oldState.missedFrames + 1;
                    
                    if (newMissedFrames <= MAX_MISS_FRAMES) {
                        // Keep the track alive with the last known position.
                        const currentState = {
                            ...oldState,
                            missedFrames: newMissedFrames,
                        };
                        NEXT_FRAME_TRACKED_FACES.set(id, currentState);
                    }
                    // If missedFrames > MAX_MISS_FRAMES, the track is naturally dropped.
                }
            }
            
            // --- 3. NEW DETECTION (Iterate over unmatched new detections) ---
            for (const detection of unmatchedDetections) {
                const boundingBox = detection.boundingBox;
                const currentCenterFlippedX = getFlippedBoxCenter(boundingBox, canvasElement.width);

                const foundId = nextTrackingId++;
                const currentSide = currentCenterFlippedX < centerLineX ? 'left' : 'right';

                const newState = {
                    id: foundId,
                    prevCenterFlippedX: currentCenterFlippedX,
                    lastSide: currentSide,
                    isEnterCounted: false, 
                    isExitCounted: false,
                    missedFrames: 0,
                    boundingBox: boundingBox,
                };
                
                NEXT_FRAME_TRACKED_FACES.set(foundId, newState);
            }
            
            // Update the global tracking map
            trackedFaceStates = NEXT_FRAME_TRACKED_FACES;
            
            // --- 4. DRAWING (Iterate over all active tracks, including ghosted ones) ---
            let actualDetectionsCount = 0; 

            for (const state of trackedFaceStates.values()) {
                const boundingBox = state.boundingBox;
                
                // Calculate flipped X for drawing
                const flippedX = canvasElement.width - boundingBox.originX - boundingBox.width;
                
                // Determine color based on persistence status
                if (state.missedFrames === 0) {
                    canvasCtx.strokeStyle = 'rgb(59, 130, 246)'; // Bright Blue (Actively detected)
                    actualDetectionsCount++;
                } else {
                    canvasCtx.strokeStyle = 'rgba(107, 114, 128, 0.5)'; // Faded Gray (Ghosted/Persistent)
                }
                
                canvasCtx.lineWidth = 4;
                
                // Draw the rounded bounding box
                const radius = 12;
                canvasCtx.beginPath();
                canvasCtx.moveTo(flippedX + radius, boundingBox.originY);
                canvasCtx.lineTo(flippedX + boundingBox.width - radius, boundingBox.originY);
                canvasCtx.arcTo(flippedX + boundingBox.width, boundingBox.originY, flippedX + boundingBox.width, boundingBox.originY + radius, radius);
                canvasCtx.lineTo(flippedX + boundingBox.width, boundingBox.originY + boundingBox.height - radius);
                canvasCtx.arcTo(flippedX + boundingBox.width, boundingBox.originY + boundingBox.height, flippedX + boundingBox.width - radius, boundingBox.originY + boundingBox.height, radius);
                canvasCtx.lineTo(flippedX + radius, boundingBox.originY + boundingBox.height);
                canvasCtx.arcTo(flippedX, boundingBox.originY + boundingBox.height, flippedX, boundingBox.originY + boundingBox.height - radius, radius);
                canvasCtx.lineTo(flippedX, boundingBox.originY + radius);
                canvasCtx.arcTo(flippedX, boundingBox.originY, flippedX + radius, boundingBox.originY, radius);
                canvasCtx.stroke();
            }

            // Update the UI counts
            enterCountElement.textContent = enterCount;
            exitCountElement.textContent = exitCount;
            // Display actual detections vs. total tracked (including persistent ones)
            detectionCount.textContent = `${actualDetectionsCount} detected (${trackedFaceStates.size} tracked)`; 

            // Loop
            window.requestAnimationFrame(detectFaces);
        }

        // Start the application setup
        createFaceDetector();