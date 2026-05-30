import React, { useCallback, useEffect, useRef, useState } from 'react';
import DetectionPanel from './DetectionPanel.jsx';
import { DownloadIcon, PauseIcon, PlayIcon, UploadIcon } from './Icons.jsx';
import { analyzeFrame } from '../services/api.js';
import { drawFaces } from '../utils/drawing.js';

const VIDEO_FRAME_INTERVAL_MS =
  parseInt(import.meta.env.VITE_VIDEO_FRAME_INTERVAL_MS, 10) || 1000;
const RECORDING_FPS = 30;

const MIME_TYPES = [
  'video/webm;codecs=vp9,opus',
  'video/webm;codecs=vp8,opus',
  'video/webm',
  'video/mp4',
];

function formatTime(seconds) {
  if (!Number.isFinite(seconds)) return '--:--';
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return '--';
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function outputExtension(mimeType) {
  return mimeType.includes('mp4') ? 'mp4' : 'webm';
}

function baseName(filename) {
  return filename.replace(/\.[^/.]+$/, '') || 'video';
}

export default function VideoUploadView({ push }) {
  const inputRef = useRef(null);
  const sourceVideoRef = useRef(null);
  const renderCanvasRef = useRef(null);
  const captureCanvasRef = useRef(null);
  const animationFrameRef = useRef(null);
  const recorderRef = useRef(null);
  const recordingStreamRef = useRef(null);
  const sourceCaptureStreamRef = useRef(null);
  const inputUrlRef = useRef(null);
  const outputUrlRef = useRef(null);
  const runIdRef = useRef(0);
  const processingRef = useRef(false);
  const analysisRef = useRef(null);
  const inferenceInFlightRef = useRef(false);
  const nextInferenceTimeRef = useRef(0);
  const lastProgressUpdateRef = useRef(0);

  const [file, setFile] = useState(null);
  const [inputUrl, setInputUrl] = useState(null);
  const [outputUrl, setOutputUrl] = useState(null);
  const [outputName, setOutputName] = useState('');
  const [status, setStatus] = useState('empty');
  const [duration, setDuration] = useState(0);
  const [progress, setProgress] = useState(0);
  const [analysis, setAnalysis] = useState(null);
  const [analyzedFrames, setAnalyzedFrames] = useState(0);
  const [lastLatency, setLastLatency] = useState(null);
  const [pipelineOnline, setPipelineOnline] = useState(null);
  const [audioIncluded, setAudioIncluded] = useState(false);
  const [error, setError] = useState('');

  const isProcessing = status === 'processing' || status === 'finalizing';

  const replaceInputUrl = useCallback((url) => {
    if (inputUrlRef.current) URL.revokeObjectURL(inputUrlRef.current);
    inputUrlRef.current = url;
    setInputUrl(url);
  }, []);

  const replaceOutputUrl = useCallback((url) => {
    if (outputUrlRef.current) URL.revokeObjectURL(outputUrlRef.current);
    outputUrlRef.current = url;
    setOutputUrl(url);
  }, []);

  const stopRecordingTracks = useCallback(() => {
    recordingStreamRef.current?.getTracks().forEach((track) => track.stop());
    sourceCaptureStreamRef.current?.getTracks().forEach((track) => track.stop());
    recordingStreamRef.current = null;
    sourceCaptureStreamRef.current = null;
  }, []);

  const stopAnimation = useCallback(() => {
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
  }, []);

  const captureSourceFrame = useCallback(() => {
    return new Promise((resolve) => {
      const video = sourceVideoRef.current;
      if (!video?.videoWidth) {
        resolve(null);
        return;
      }

      const canvas = captureCanvasRef.current || document.createElement('canvas');
      captureCanvasRef.current = canvas;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
      canvas.toBlob((blob) => resolve(blob), 'image/jpeg', 0.85);
    });
  }, []);

  const analyzeCurrentFrame = useCallback(async () => {
    inferenceInFlightRef.current = true;
    const startedAt = performance.now();

    try {
      const blob = await captureSourceFrame();
      if (!blob || !processingRef.current) return;

      const data = await analyzeFrame(blob);
      if (!processingRef.current) return;

      analysisRef.current = data;
      setAnalysis(data);
      setAnalyzedFrames((count) => count + 1);
      setLastLatency(Math.round(performance.now() - startedAt));
      setPipelineOnline(true);
      setError('');
    } catch (requestError) {
      if (!processingRef.current) return;
      setPipelineOnline(false);
      setError(`Pipeline request failed: ${requestError.message}`);
    } finally {
      inferenceInFlightRef.current = false;
    }
  }, [captureSourceFrame]);

  const renderFrame = useCallback(function paintFrame() {
    if (!processingRef.current) return;

    const video = sourceVideoRef.current;
    const canvas = renderCanvasRef.current;
    const ctx = canvas?.getContext('2d');

    if (video && ctx && video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      const latestAnalysis = analysisRef.current;
      if (latestAnalysis) {
        const annotationScale = Math.max(1, canvas.width / 960);
        drawFaces(
          ctx,
          latestAnalysis.faces,
          latestAnalysis.image_width,
          latestAnalysis.image_height,
          false,
          false,
          annotationScale,
        );
      }

      if (
        !inferenceInFlightRef.current &&
        video.currentTime >= nextInferenceTimeRef.current
      ) {
        nextInferenceTimeRef.current =
          video.currentTime + VIDEO_FRAME_INTERVAL_MS / 1000;
        void analyzeCurrentFrame();
      }

      const now = performance.now();
      if (now - lastProgressUpdateRef.current >= 200) {
        lastProgressUpdateRef.current = now;
        setProgress(duration ? Math.min(video.currentTime / duration, 1) : 0);
      }
    }

    animationFrameRef.current = requestAnimationFrame(paintFrame);
  }, [analyzeCurrentFrame, duration]);

  const finalizeRecording = useCallback(() => {
    if (!processingRef.current) return;

    processingRef.current = false;
    stopAnimation();
    setProgress(1);
    setStatus('finalizing');

    const recorder = recorderRef.current;
    if (recorder && recorder.state !== 'inactive') {
      recorder.stop();
    } else {
      stopRecordingTracks();
      setStatus('error');
      setError('The browser could not finish the annotated recording.');
    }
  }, [stopAnimation, stopRecordingTracks]);

  const cancelProcessing = useCallback((showToast = true) => {
    if (!processingRef.current && status !== 'finalizing') return;

    runIdRef.current += 1;
    processingRef.current = false;
    sourceVideoRef.current?.pause();
    stopAnimation();

    const recorder = recorderRef.current;
    if (recorder && recorder.state !== 'inactive') {
      recorder.stop();
    } else {
      stopRecordingTracks();
    }

    analysisRef.current = null;
    setAnalysis(null);
    setProgress(0);
    setStatus(file ? 'ready' : 'empty');
    if (showToast) push('Video analysis cancelled', 'warn');
  }, [file, push, status, stopAnimation, stopRecordingTracks]);

  const resetOutput = useCallback(() => {
    replaceOutputUrl(null);
    setOutputName('');
    setAudioIncluded(false);
  }, [replaceOutputUrl]);

  const selectVideo = useCallback((selectedFile) => {
    if (!selectedFile) return;
    if (!selectedFile.type.startsWith('video/')) {
      push('Choose a valid video file', 'error');
      return;
    }

    cancelProcessing(false);
    resetOutput();
    analysisRef.current = null;
    setAnalysis(null);
    setAnalyzedFrames(0);
    setLastLatency(null);
    setPipelineOnline(null);
    setDuration(0);
    setProgress(0);
    setError('');
    setFile(selectedFile);
    setStatus('loading');
    replaceInputUrl(URL.createObjectURL(selectedFile));
  }, [cancelProcessing, push, replaceInputUrl, resetOutput]);

  const startProcessing = useCallback(async () => {
    const video = sourceVideoRef.current;
    const canvas = renderCanvasRef.current;

    if (!file || !video?.videoWidth || !canvas) return;
    if (!canvas.captureStream || typeof MediaRecorder === 'undefined') {
      setStatus('error');
      setError('This browser cannot record an annotated canvas. Use a recent Chrome, Edge, or Firefox release.');
      return;
    }

    resetOutput();
    analysisRef.current = null;
    setAnalysis(null);
    setAnalyzedFrames(0);
    setLastLatency(null);
    setPipelineOnline(null);
    setProgress(0);
    setError('');

    video.pause();
    video.currentTime = 0;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);

    let recordingStream;
    try {
      recordingStream = canvas.captureStream(RECORDING_FPS);
    } catch (streamError) {
      setStatus('error');
      setError(`Could not start video recording: ${streamError.message}`);
      return;
    }
    recordingStreamRef.current = recordingStream;

    let captureStream = null;
    try {
      captureStream = video.captureStream?.() || video.mozCaptureStream?.() || null;
    } catch {
      // Audio capture is optional. The annotated video can still be recorded.
    }
    sourceCaptureStreamRef.current = captureStream;
    const audioTracks = captureStream?.getAudioTracks() || [];
    audioTracks.forEach((track) => recordingStream.addTrack(track));
    setAudioIncluded(audioTracks.length > 0);

    const mimeType = typeof MediaRecorder.isTypeSupported === 'function'
      ? MIME_TYPES.find((type) => MediaRecorder.isTypeSupported(type))
      : null;
    let recorder;
    try {
      recorder = new MediaRecorder(
        recordingStream,
        mimeType ? { mimeType } : undefined,
      );
    } catch (recorderError) {
      recordingStream.getTracks().forEach((track) => track.stop());
      captureStream?.getTracks().forEach((track) => track.stop());
      recordingStreamRef.current = null;
      sourceCaptureStreamRef.current = null;
      setStatus('error');
      setError(`Could not create the annotated recording: ${recorderError.message}`);
      return;
    }
    const runId = ++runIdRef.current;
    recorderRef.current = recorder;
    const chunks = [];

    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) chunks.push(event.data);
    };
    recorder.onstop = () => {
      if (recorderRef.current === recorder) recorderRef.current = null;
      recordingStream.getTracks().forEach((track) => track.stop());
      captureStream?.getTracks().forEach((track) => track.stop());
      if (recordingStreamRef.current === recordingStream) recordingStreamRef.current = null;
      if (sourceCaptureStreamRef.current === captureStream) sourceCaptureStreamRef.current = null;
      if (runId !== runIdRef.current) return;

      const blob = new Blob(chunks, { type: recorder.mimeType || 'video/webm' });
      if (!blob.size) {
        setStatus('error');
        setError('The annotated recording was empty. Try the video again in a recent browser.');
        return;
      }

      const extension = outputExtension(blob.type);
      replaceOutputUrl(URL.createObjectURL(blob));
      setOutputName(`${baseName(file.name)}-annotated.${extension}`);
      setStatus('complete');
      push('Annotated video is ready to download', 'success');
    };

    processingRef.current = true;
    nextInferenceTimeRef.current = 0;
    lastProgressUpdateRef.current = 0;
    setStatus('processing');
    recorder.start(1000);

    try {
      await video.play();
      animationFrameRef.current = requestAnimationFrame(renderFrame);
    } catch (playError) {
      runIdRef.current += 1;
      processingRef.current = false;
      recorder.stop();
      setStatus('error');
      setError(`Could not play this video: ${playError.message}`);
    }
  }, [file, push, renderFrame, replaceOutputUrl, resetOutput, stopRecordingTracks]);

  useEffect(() => {
    return () => {
      runIdRef.current += 1;
      processingRef.current = false;
      stopAnimation();
      const recorder = recorderRef.current;
      if (recorder && recorder.state !== 'inactive') recorder.stop();
      else stopRecordingTracks();
      if (inputUrlRef.current) URL.revokeObjectURL(inputUrlRef.current);
      if (outputUrlRef.current) URL.revokeObjectURL(outputUrlRef.current);
    };
  }, [stopAnimation, stopRecordingTracks]);

  const onFileInput = (event) => {
    selectVideo(event.target.files?.[0]);
    event.target.value = '';
  };

  const onDrop = (event) => {
    event.preventDefault();
    selectVideo(event.dataTransfer.files?.[0]);
  };

  const statusCopy = {
    empty: 'Choose a local video to begin',
    loading: 'Reading video metadata',
    ready: 'Ready to analyze in real time',
    processing: 'Analyzing and recording annotated output',
    finalizing: 'Finalizing downloadable video',
    complete: 'Annotated video ready',
    error: 'Video processor needs attention',
  }[status];

  return (
    <main className="video-upload">
      <section className="video-upload__stage-col">
        <header className="camera__head">
          <div className="camera__head-title">
            <span className="camera__section-num">§ 01</span>
            <span className="camera__section-title">Uploaded Video</span>
          </div>
          <span className="t-label">Real-time annotated export</span>
        </header>

        <div
          className={`camera__viewport video-upload__viewport ${!file ? 'video-upload__viewport--empty' : ''}`}
          onDragOver={(event) => event.preventDefault()}
          onDrop={onDrop}
        >
          <span className="camera__corner tr" />
          <span className="camera__corner bl" />

          <video
            ref={sourceVideoRef}
            className="video-upload__media"
            src={inputUrl || undefined}
            controls={Boolean(file) && !isProcessing}
            preload="metadata"
            hidden={!inputUrl || isProcessing || Boolean(outputUrl)}
            onLoadedMetadata={(event) => {
              setDuration(event.currentTarget.duration);
              setStatus('ready');
            }}
            onEnded={finalizeRecording}
            onError={() => {
              if (!inputUrl) return;
              setStatus('error');
              setError('The browser could not read this video file.');
            }}
          />

          <canvas
            ref={renderCanvasRef}
            className="video-upload__media"
            hidden={!isProcessing}
          />

          {outputUrl && status === 'complete' && (
            <video
              className="video-upload__media"
              src={outputUrl}
              controls
              preload="metadata"
            />
          )}

          {!file && (
            <button
              className="video-upload__drop"
              onClick={() => inputRef.current?.click()}
              type="button"
            >
              <UploadIcon className="video-upload__drop-icon" />
              <span className="video-upload__drop-title">Drop a video here</span>
              <span className="video-upload__drop-sub">or choose a file from your device</span>
            </button>
          )}

          {status === 'processing' && (
            <>
              <div className="camera__rec">PROCESSING</div>
              <div className="camera__timecode">
                {formatTime((sourceVideoRef.current?.currentTime || 0))} / {formatTime(duration)}
              </div>
            </>
          )}
        </div>

        <div className="video-upload__controls">
          <div className="camera__controls-buttons">
            <button
              className="btn btn--ghost"
              onClick={() => inputRef.current?.click()}
              type="button"
            >
              <UploadIcon />
              {file ? 'Choose another' : 'Choose video'}
            </button>
            {!isProcessing && (
              <button
                className="btn btn--primary"
                onClick={startProcessing}
                disabled={!file || status === 'loading'}
                type="button"
              >
                <PlayIcon />
                {status === 'complete' ? 'Analyze again' : 'Analyze video'}
              </button>
            )}
            {isProcessing && (
              <button
                className="btn btn--danger-ghost"
                onClick={() => cancelProcessing()}
                type="button"
              >
                <PauseIcon />
                Cancel analysis
              </button>
            )}
            {outputUrl && status === 'complete' && (
              <a className="btn btn--accent" href={outputUrl} download={outputName}>
                <DownloadIcon />
                Download video
              </a>
            )}
          </div>
          <span className={`video-upload__status ${status === 'error' ? 'stop' : ''}`}>
            {statusCopy}
          </span>
        </div>

        <div className="video-upload__progress">
          <span
            className="video-upload__progress-fill"
            style={{ width: `${progress * 100}%` }}
          />
        </div>

        {error && <div className="video-upload__error">{error}</div>}

        <input
          ref={inputRef}
          className="video-upload__input"
          type="file"
          accept="video/*"
          onChange={onFileInput}
        />
      </section>

      <aside className="column video-upload__aside">
        <DetectionPanel analysis={analysis} />

        <section className="section">
          <header className="section__head">
            <div className="section__head-title">
              <span className="section__num">§ 03</span>
              <span className="section__title">Processing Details</span>
            </div>
            <span className="section__aside">Video pipeline</span>
          </header>

          <dl className="video-upload__details">
            <div className="video-upload__detail">
              <dt>Source</dt>
              <dd>{file?.name || 'No file selected'}</dd>
            </div>
            <div className="video-upload__detail">
              <dt>File size</dt>
              <dd>{file ? formatBytes(file.size) : '--'}</dd>
            </div>
            <div className="video-upload__detail">
              <dt>Duration</dt>
              <dd>{file ? formatTime(duration) : '--:--'}</dd>
            </div>
            <div className="video-upload__detail">
              <dt>Frames analyzed</dt>
              <dd>{String(analyzedFrames).padStart(2, '0')}</dd>
            </div>
            <div className="video-upload__detail">
              <dt>Latest pipeline latency</dt>
              <dd>{lastLatency == null ? '--' : `${lastLatency} ms`}</dd>
            </div>
            <div className="video-upload__detail">
              <dt>Pipeline</dt>
              <dd className={pipelineOnline === false ? 'stop' : pipelineOnline ? 'go' : ''}>
                {pipelineOnline == null ? 'Standby' : pipelineOnline ? 'Online' : 'Offline'}
              </dd>
            </div>
            <div className="video-upload__detail">
              <dt>Inference sample rate</dt>
              <dd>Every {VIDEO_FRAME_INTERVAL_MS} ms</dd>
            </div>
            <div className="video-upload__detail">
              <dt>Output audio</dt>
              <dd>{audioIncluded ? 'Included' : 'Video only'}</dd>
            </div>
          </dl>
        </section>
      </aside>
    </main>
  );
}
