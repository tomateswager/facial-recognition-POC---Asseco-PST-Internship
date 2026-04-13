/* ==== frontend/js/smart-camera.js ==== */

class SmartCamera {
    constructor(containerId, options = {}) {
        this.container = document.getElementById(containerId);
        this.imageBox = this.container.querySelector('.image-box');
        this.video = this.container.querySelector('video');

        let canvas = this.container.querySelector('.overlay-canvas');
        if (!canvas) {
            canvas = document.createElement('canvas');
            canvas.className = 'overlay-canvas';
            canvas.style.position = 'absolute';
            canvas.style.top = '0';
            canvas.style.left = '0';
            canvas.style.width = '100%';
            canvas.style.height = '100%';
            canvas.style.zIndex = '10';
            canvas.style.pointerEvents = 'none';
            this.container.querySelector('.image-box').appendChild(canvas);
        }
        this.canvas = canvas;

        this.captureCanvas = document.createElement('canvas');
        this.feedbackText = this._createFeedbackText();

        this.options = {
            autoCapture: true,
            onCapture: options.onCapture || (() => {}),
            mode: options.mode || 'live',
            // Analyze stream over time with a motion challenge before capture.
            requireChallenge: options.requireChallenge || false,
        };

        this.stream = null;
        this.isDetecting = false;
        this.modelsLoaded = false;
        this.videoReady = false;
        this.resetLivenessState();
    }

    _createFeedbackText() {
        let div = this.container.querySelector('.camera-feedback');
        if (!div) {
            div = document.createElement('div');
            div.className = 'camera-feedback';
            div.style.display = 'none';
            div.style.zIndex = '20';
            this.container.querySelector('.image-box').appendChild(div);
        }
        return div;
    }

    resetLivenessState() {
        this.preChallengeStableCount = 0;
        this.challengeActive = false;
        this.challengePassed = false;
        this.challengeDirection = null;
        this.challengeBaselineX = null;
        this.faceStableCount = 0;
        this.lastFaceX = null;
        this.lastFaceY = null;
    }

    startMovementChallenge(faceX) {
        this.challengeActive = true;
        this.challengePassed = false;
        this.challengeDirection = 'either';
        this.challengeBaselineX = faceX;
        this.faceStableCount = 0;
    }

    async start() {
        this.showFeedback('A iniciar...', 'info');
        this.isDetecting = true;
        this.resetLivenessState();

        this.startDrawingLoop();
        this.loadAI();

        try {
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
            });
            this.video.srcObject = this.stream;
            this.video.style.display = 'block';
            this.video.oncanplay = () => { this.videoReady = true; };
            await this.video.play();
        } catch (err) {
            console.error(err);
            this.showFeedback('Erro: Camera bloqueada', 'error');
        }
    }

    async loadAI() {
        try {
            if (typeof faceapi === 'undefined') {
                throw new Error('faceapi_not_loaded');
            }
            await faceapi.nets.tinyFaceDetector.loadFromUri('/faceapi-models');
            this.modelsLoaded = true;
        } catch (e) {
            console.warn('IA offline. Usando modo temporizador.');
            setTimeout(() => this.runFallbackTimer(), 3000);
        }
    }

    startDrawingLoop() {
        const ctx = this.canvas.getContext('2d');

        const loop = async () => {
            if (!this.isDetecting) return;

            const drawHost = this.imageBox || this.container;
            const w = drawHost.clientWidth || this.container.offsetWidth;
            const h = drawHost.clientHeight || this.container.offsetHeight;

            if (this.canvas.width !== w || this.canvas.height !== h) {
                this.canvas.width = w;
                this.canvas.height = h;
            }

            let ovalColor = 'white';
            let message = 'Coloque o rosto na oval';
            let type = 'info';
            const guide = this.getGuideOval(w, h);

            if (this.modelsLoaded && this.videoReady && this.video.videoWidth > 0) {
                try {
                    const displaySize = { width: w, height: h };
                    const detections = await faceapi.detectAllFaces(
                        this.video,
                        new faceapi.TinyFaceDetectorOptions({ inputSize: 320, scoreThreshold: 0.4 })
                    );
                    const resized = faceapi.resizeResults(detections, displaySize);

                    if (resized.length === 0) {
                        message = 'Aproxime o rosto...';
                        type = 'warning';
                        this.resetLivenessState();
                    } else if (resized.length > 1) {
                        message = 'Apenas uma pessoa!';
                        type = 'error';
                        this.resetLivenessState();
                    } else {
                        const box = resized[0].box || resized[0].detection?.box;
                        const faceX = box.x + box.width / 2;
                        const faceY = box.y + box.height / 2;

                        const dist = Math.sqrt(
                            Math.pow(faceX - guide.centerX, 2) + Math.pow(faceY - guide.centerY, 2)
                        );
                        const centerTolerance = guide.ovalW * 0.18;
                        const isCentered = dist < centerTolerance;

                        const isSizeGood = box.height > guide.ovalH * 0.38 && box.height < guide.ovalH * 0.95;
                        const isPositionGood = isCentered && isSizeGood;

                        if (!isPositionGood) {
                            ovalColor = '#ffaa00';
                            message = !isSizeGood
                                ? (box.height < guide.ovalH * 0.38 ? 'Aproxime-se' : 'Afaste-se um pouco')
                                : 'Centre o rosto';
                            type = 'warning';
                            this.preChallengeStableCount = 0;
                            this.faceStableCount = 0;
                            if (!this.challengePassed) {
                                this.challengeActive = false;
                                this.challengeDirection = null;
                                this.challengeBaselineX = null;
                            }
                        } else if (!this.options.requireChallenge) {
                            ovalColor = '#00ff00';
                            type = 'success';

                            // Verificar se o rosto está parado entre frames
                            const moveThreshold = guide.ovalW * 0.03; // 3% da largura da oval
                            if (this.lastFaceX !== null) {
                                const moved = Math.sqrt(
                                    Math.pow(faceX - this.lastFaceX, 2) +
                                    Math.pow(faceY - this.lastFaceY, 2)
                                );
                                if (moved < moveThreshold) {
                                    this.faceStableCount++;
                                } else {
                                    this.faceStableCount = 0;
                                }
                            }
                            this.lastFaceX = faceX;
                            this.lastFaceY = faceY;

                            const framesLeft = Math.max(0, 20 - this.faceStableCount);
                            message = framesLeft > 0 ? `Não se mexa... (${framesLeft})` : 'A capturar...';

                            if (this.faceStableCount > 20) {
                                this.doCapture();
                                return;
                            }
                        } else if (!this.challengeActive && !this.challengePassed) {
                            ovalColor = '#00ff00';
                            message = 'Mantenha-se estavel...';
                            type = 'success';
                            this.preChallengeStableCount++;

                            if (this.preChallengeStableCount > 6) {
                                this.startMovementChallenge(faceX);
                            }
                        } else if (this.challengeActive && !this.challengePassed) {
                            const moveThreshold = guide.ovalW * 0.08;
                            const deltaX = faceX - this.challengeBaselineX;
                            const movedEnough = Math.abs(deltaX) > moveThreshold;

                            ovalColor = '#ffaa00';
                            message = 'Prova de vida: mova a cabeca para um dos lados';
                            type = 'warning';

                            if (movedEnough) {
                                this.challengeActive = false;
                                this.challengePassed = true;
                                this.faceStableCount = 0;
                            }
                        } else {
                            ovalColor = '#00ff00';
                            message = 'Prova de vida ok. Volte ao centro e nao se mexa...';
                            type = 'success';
                            this.faceStableCount++;

                            if (this.faceStableCount > 12) {
                                this.doCapture();
                                return;
                            }
                        }
                    }
                } catch (e) {
                    console.warn('[SmartCamera] Detection error:', e.message || e);
                }
            }

            this.drawOval(ctx, w, h, ovalColor, guide);
            this.showFeedback(message, type);
            requestAnimationFrame(loop);
        };

        loop();
    }

    getGuideOval(width, height) {
        return {
            centerX: width / 2,
            centerY: height * 0.47,
            ovalW: width * 0.42,
            ovalH: height * 0.82,
        };
    }

    drawOval(ctx, width, height, color, guide = this.getGuideOval(width, height)) {
        ctx.clearRect(0, 0, width, height);

        ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        ctx.fillRect(0, 0, width, height);

        const ovalW = guide.ovalW;
        const ovalH = guide.ovalH;
        const centerX = guide.centerX;
        const centerY = guide.centerY;

        ctx.globalCompositeOperation = 'destination-out';
        ctx.beginPath();
        ctx.ellipse(centerX, centerY, ovalW / 2, ovalH / 2, 0, 0, 2 * Math.PI);
        ctx.fill();

        ctx.globalCompositeOperation = 'source-over';
        ctx.strokeStyle = color;
        ctx.lineWidth = 5;
        ctx.beginPath();
        ctx.ellipse(centerX, centerY, ovalW / 2, ovalH / 2, 0, 0, 2 * Math.PI);
        ctx.stroke();
    }

    runFallbackTimer() {
        if (!this.isDetecting) return;

        // Mostrar botão de captura manual (mais fiável que timer cego)
        this.showFeedback('IA indisponível — posicione-se e clique em Capturar', 'warning');

        const existingBtn = this.container.querySelector('.smart-cam-capture-btn');
        if (existingBtn) existingBtn.remove();

        const btn = document.createElement('button');
        btn.className = 'btn btn-success smart-cam-capture-btn';
        btn.style.cssText = 'position:absolute;bottom:60px;left:50%;transform:translateX(-50%);z-index:30;';
        btn.textContent = '📸 Capturar';
        btn.onclick = () => {
            btn.remove();
            this.doCapture();
        };
        this.container.querySelector('.image-box').appendChild(btn);
    }

    async doCapture() {
        this.isDetecting = false;
        this.showFeedback('A processar...', 'success');

        try {
            const ctx = this.captureCanvas.getContext('2d');
            if (!ctx) throw new Error('capture_ctx_unavailable');

            const videoW = this.video.videoWidth || 0;
            const videoH = this.video.videoHeight || 0;
            if (!videoW || !videoH) {
                throw new Error('video_dimensions_unavailable');
            }

            this.captureCanvas.width = videoW;
            this.captureCanvas.height = videoH;
            ctx.drawImage(this.video, 0, 0, this.captureCanvas.width, this.captureCanvas.height);

            this.captureCanvas.toBlob(blob => {
                const livenessPassedAtCapture = this.challengePassed;
                const livenessMethodAtCapture = this.options.requireChallenge ? 'head_turn' : null;

                try {
                    this.stop();
                } catch (e) {
                    console.error('Erro ao parar camera apos captura:', e);
                }

                if (!blob) {
                    this.showFeedback('Falha ao capturar. Tente novamente.', 'error');
                    return;
                }

                const meta = {
                    livenessPassed: this.options.requireChallenge ? livenessPassedAtCapture : false,
                    livenessMethod: livenessMethodAtCapture,
                };
                this.options.onCapture(blob, meta);
            }, 'image/jpeg', 0.95);
        } catch (e) {
            console.error('Erro no doCapture:', e);
            this.showFeedback('Falha na captura. Tente novamente.', 'error');
            setTimeout(() => this.stop(), 700);
        }
    }

    stop() {
        if (this.stream) {
            this.stream.getTracks().forEach(t => t.stop());
            this.stream = null;
        }

        this.isDetecting = false;
        this.videoReady = false;
        this.video.style.display = 'none';
        const ctx = this.canvas.getContext('2d');
        ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.hideFeedback();
        this.resetLivenessState();

        const btn = this.container.querySelector('.smart-cam-capture-btn');
        if (btn) btn.remove();
    }

    showFeedback(text, type) {
        if (this.feedbackText.textContent !== text) {
            this.feedbackText.textContent = text;
            this.feedbackText.className = `camera-feedback feedback-${type}`;
            this.feedbackText.style.display = 'block';
        }
    }

    hideFeedback() {
        if (this.feedbackText) {
            this.feedbackText.style.display = 'none';
        }
    }
}
