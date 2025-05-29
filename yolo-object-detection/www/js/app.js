const { useState, useEffect, useRef } = React;

const App = () => {
    const [cameraMode, setCameraMode] = useState('environment'); // 'environment' (후면) 또는 'user' (전면)
    const [detections, setDetections] = useState([]);
    const [isProcessing, setIsProcessing] = useState(false);
    const [errorMessage, setErrorMessage] = useState('');
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const streamRef = useRef(null);
    const timerRef = useRef(null);

    useEffect(() => {
        startCamera();
        return () => {
            stopCamera();
            if (timerRef.current) {
                clearInterval(timerRef.current);
            }
        };
    }, [cameraMode]);

    const startCamera = async () => {
        try {
            if (streamRef.current) {
                stopCamera();
            }

            const constraints = {
                video: {
                    facingMode: cameraMode
                }
            };

            const stream = await navigator.mediaDevices.getUserMedia(constraints);
            streamRef.current = stream;

            if (videoRef.current) {
                videoRef.current.srcObject = stream;
                videoRef.current.play();

                // 비디오 크기에 맞게 캔버스 설정
                videoRef.current.onloadedmetadata = () => {
                    if (canvasRef.current) {
                        canvasRef.current.width = videoRef.current.videoWidth;
                        canvasRef.current.height = videoRef.current.videoHeight;
                    }

                    // 주기적으로 객체 감지 실행
                    timerRef.current = setInterval(() => {
                        if (!isProcessing) {
                            detectObjects();
                        }
                    }, 10000); // 10초마다
                };
            }
        } catch (err) {
            setErrorMessage(`카메라 접근 오류: ${err.message}`);
            console.error('카메라 접근 오류:', err);
        }
    };

    const stopCamera = () => {
        if (streamRef.current) {
            streamRef.current.getTracks().forEach(track => track.stop());
            streamRef.current = null;
        }
    };

    const switchCamera = () => {
        setCameraMode(prev => prev === 'environment' ? 'user' : 'environment');
    };

    const captureImage = () => {
        if (!videoRef.current || !canvasRef.current) return null;

        const canvas = document.createElement('canvas');
        canvas.width = videoRef.current.videoWidth;
        canvas.height = videoRef.current.videoHeight;

        const ctx = canvas.getContext('2d');
        ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);

        return canvas.toDataURL('image/jpeg');
    };

    const detectObjects = async () => {
        if (!videoRef.current || !canvasRef.current) return;

        try {
            setIsProcessing(true);
            const imageData = captureImage();

            if (!imageData) {
                console.error('이미지 캡처 실패');
                return;
            }

            const response = await fetch('/api/detect', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ image: imageData }),
            });

            const result = await response.json();

            if (result.success) {
                console.log('감지 결과:', result.detections);
                setDetections(result.detections);
                drawDetections(result.detections);
            } else {
                console.error('객체 감지 오류:', result.error);
                setErrorMessage(`객체 감지 오류: ${result.error}`);
            }
        } catch (err) {
            console.error('API 요청 오류:', err);
            setErrorMessage(`API 요청 오류: ${err.message}`);
        } finally {
            setIsProcessing(false);
        }
    };

    const drawDetections = (detectionResults) => {
        if (!canvasRef.current || !videoRef.current) return;

        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');

        // 캔버스 초기화
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // 감지된 객체 그리기
        const scaleX = canvas.width / videoRef.current.videoWidth;
        const scaleY = canvas.height / videoRef.current.videoHeight;

        detectionResults.forEach(detection => {
            const [x1, y1, x2, y2] = detection.bbox;
            const scaledX1 = x1 * scaleX;
            const scaledY1 = y1 * scaleY;
            const scaledWidth = (x2 - x1) * scaleX;
            const scaledHeight = (y2 - y1) * scaleY;

            // 경계 상자 그리기
            ctx.strokeStyle = '#00FF00';
            ctx.lineWidth = 2;
            ctx.strokeRect(scaledX1, scaledY1, scaledWidth, scaledHeight);

            // 텍스트 배경 그리기
            const label = `${detection.name} ${Math.round(detection.confidence * 100)}%`;
            const textWidth = ctx.measureText(label).width;
            ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
            ctx.fillRect(scaledX1, scaledY1 - 20, textWidth + 10, 20);

            // 텍스트 그리기
            ctx.fillStyle = '#FFFFFF';
            ctx.font = '16px Arial';
            ctx.fillText(label, scaledX1 + 5, scaledY1 - 5);
        });
    };

    const renderDetectionList = () => {
        if (detections.length === 0) {
            return <p>감지된 객체가 없습니다.</p>;
        }

        // 클래스별로 그룹화
        const groupedDetections = {};
        detections.forEach(detection => {
            const name = detection.name;
            if (!groupedDetections[name]) {
                groupedDetections[name] = 0;
            }
            groupedDetections[name]++;
        });

        return (
            <div>
                <h5>감지된 객체</h5>
                {Object.entries(groupedDetections).map(([name, count], index) => (
                    <div key={index} className="detection-item">
                        <span>{name}</span>
                        <span className="badge bg-primary">{count}</span>
                    </div>
                ))}
                <p className="mt-2">총 {detections.length}개 객체 감지됨</p>
            </div>
        );
    };

    return (
        <div className="container">
            <h1 className="text-center mb-4">YOLOv8 객체 인식</h1>

            {errorMessage && (
                <div className="alert alert-danger" role="alert">
                    {errorMessage}
                </div>
            )}

            <div className="camera-container">
                <video ref={videoRef} id="videoElement" autoPlay playsInline></video>
                <canvas ref={canvasRef} id="canvasElement"></canvas>
            </div>

            <div className="controls mt-3">
                <button
                    className="btn btn-primary"
                    onClick={switchCamera}
                >
                    {cameraMode === 'environment' ? '전면 카메라로 전환' : '후면 카메라로 전환'}
                </button>

                <button
                    className="btn btn-success"
                    onClick={detectObjects}
                    disabled={isProcessing}
                >
                    객체 감지
                    {isProcessing && (
                        <span className="spinner-border spinner-border-sm ms-2" role="status" aria-hidden="true"></span>
                    )}
                </button>
            </div>

            <div className="detection-list mt-3">
                {renderDetectionList()}
            </div>
        </div>
    );
};

ReactDOM.render(<App />, document.getElementById('app'));