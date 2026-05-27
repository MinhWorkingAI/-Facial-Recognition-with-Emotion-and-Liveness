/**
 * Draw bounding boxes + per-face stats on the overlay canvas of the main page.
 *
 * Each detected face is annotated with:
 *  - Corner-bracket bounding box 
 *  - 5 landmark keypoints
 *  - Identity label above the box
 *  - Emotion + liveness chip below the box
 *  - Detection confidence at the bottom-right corner of the box
 */

const PAPER       = '#F2EAD3';   // brighter cream for boxes
const PAPER_DIM   = '#E8DFC9';
const ACCENT      = '#E2823A';
const SIGNAL_GO   = '#8AAF5E';
const SIGNAL_STOP = '#D14A2F';
const INK         = '#1F1A16';
const BG_PLATE    = 'rgba(11, 9, 7, 0.82)';

export function clearCanvas(ctx) {
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
}

const capitalize = (s) =>
  s ? s.charAt(0).toUpperCase() + s.slice(1).toLowerCase() : '';

/**
 * @param {CanvasRenderingContext2D} ctx
 * @param {Array} faces  - from FrameAnalysisResponse.faces
 * @param {number} imgW
 * @param {number} imgH
 * @param {boolean} mirrored - whether display is CSS-mirrored
 */
export function drawFaces(ctx, faces, imgW, imgH, mirrored = true) {
  const canvas = ctx.canvas;
  clearCanvas(ctx);
  if (!faces?.length || !imgW || !imgH) return;

  // Match object-fit: cover of the video element
  const vidAspect = imgW / imgH;
  const canAspect = canvas.width / canvas.height;
  let scale, offsetX = 0, offsetY = 0;

  if (canAspect > vidAspect) {
    scale = canvas.width / imgW;
    offsetY = (canvas.height - imgH * scale) / 2;
  } else {
    scale = canvas.height / imgH;
    offsetX = (canvas.width - imgW * scale) / 2;
  }

  const project = (x, y, isNorm) => {
    const px = isNorm ? x * imgW : x;
    const py = isNorm ? y * imgH : y;
    let cx = px * scale + offsetX;
    const cy = py * scale + offsetY;
    if (mirrored) cx = canvas.width - cx;
    return { x: cx, y: cy };
  };

  for (const face of faces) {
    const bbox = face.face?.bbox;
    if (!bbox) continue;

    const isNormalized =
      bbox.x <= 1.0 && bbox.y <= 1.0 && bbox.w <= 1.0 && bbox.h <= 1.0;

    let bx, by, bw, bh;
    if (isNormalized) {
      bx = bbox.x * imgW * scale + offsetX;
      by = bbox.y * imgH * scale + offsetY;
      bw = bbox.w * imgW * scale;
      bh = bbox.h * imgH * scale;
    } else {
      bx = bbox.x * scale + offsetX;
      by = bbox.y * scale + offsetY;
      bw = bbox.w * scale;
      bh = bbox.h * scale;
    }
    if (mirrored) bx = canvas.width - bx - bw;

    const isSpoof = face.anti_spoofing?.label === 'spoof';
    const boxColor = isSpoof ? SIGNAL_STOP : PAPER;

    // Bounding box with drop shadow for visibility 
    ctx.save();
    ctx.shadowColor = 'rgba(0, 0, 0, 0.7)';
    ctx.shadowBlur = 6;
    ctx.shadowOffsetX = 0;
    ctx.shadowOffsetY = 1;

    ctx.strokeStyle = boxColor;
    ctx.lineWidth = 2.5;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    const corner = Math.min(26, bw * 0.22, bh * 0.22);
    const drawCorner = (x, y, dx1, dy1, dx2, dy2) => {
      ctx.beginPath();
      ctx.moveTo(x + dx1, y + dy1);
      ctx.lineTo(x, y);
      ctx.lineTo(x + dx2, y + dy2);
      ctx.stroke();
    };
    drawCorner(bx, by,           0, corner,  corner, 0);
    drawCorner(bx + bw, by,     -corner, 0,  0, corner);
    drawCorner(bx, by + bh,      0, -corner, corner, 0);
    drawCorner(bx + bw, by + bh,-corner, 0,  0, -corner);
    ctx.restore();

    // Landmark keypoints (5 from SCRFD) 
    const keypoints = face.face?.keypoints || [];
    if (keypoints.length > 0) {
      ctx.save();
      ctx.shadowColor = 'rgba(0, 0, 0, 0.6)';
      ctx.shadowBlur = 3;
      ctx.fillStyle = boxColor;
      for (const kp of keypoints) {
        const kpIsNorm = kp.x <= 1.0 && kp.y <= 1.0;
        const { x, y } = project(kp.x, kp.y, kpIsNorm);
        ctx.beginPath();
        ctx.arc(x, y, 2.5, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();
    }

    // Identity label (above box)
    const name = face.recognition?.label;
    const matched = face.recognition?.matched;
    if (name) {
      const text = (matched && name !== 'unknown')
        ? name.toUpperCase()
        : 'UNIDENTIFIED';
      drawLabelPlate(
        ctx,
        bx, by - 26,
        text,
        boxColor,
        matched ? INK : PAPER_DIM,
      );
    }

    // Per-face stats chip (below box) 
    drawStatsChip(ctx, bx, by + bh + 6, face, isSpoof);

    // Detection confidence (bottom-right of box) 
    const conf = face.face?.detection_confidence;
    if (typeof conf === 'number' && conf > 0) {
      ctx.save();
      ctx.shadowColor = 'rgba(0, 0, 0, 0.7)';
      ctx.shadowBlur = 3;
      ctx.font = '600 10px "IBM Plex Mono", monospace';
      ctx.fillStyle = boxColor;
      const txt = `${(conf * 100).toFixed(0)}%`;
      const tw = ctx.measureText(txt).width;
      ctx.fillText(txt, bx + bw - tw - 4, by + bh - 6);
      ctx.restore();
    }
  }
}

/**
 * Draw a single-color label on a solid background plate.
 * Returns the height of the rendered plate.
 */
function drawLabelPlate(ctx, x, y, text, bgColor, textColor) {
  ctx.font = '500 11px "IBM Plex Mono", monospace';
  const padX = 7;
  const padY = 4;
  const height = 20;
  const tw = ctx.measureText(text).width;

  // Drop shadow for the plate itself
  ctx.save();
  ctx.shadowColor = 'rgba(0, 0, 0, 0.5)';
  ctx.shadowBlur = 4;
  ctx.shadowOffsetY = 1;
  ctx.fillStyle = bgColor;
  ctx.fillRect(x, y, tw + padX * 2, height);
  ctx.restore();

  ctx.fillStyle = textColor;
  ctx.textBaseline = 'middle';
  ctx.fillText(text, x + padX, y + height / 2 + 1);
  ctx.textBaseline = 'alphabetic'; // restore default
  return height;
}

/**
 * Draw a chip beneath the bounding box showing emotion + liveness.
 * Format:  [ HAPPY · ● LIVE ]
 * The liveness pill uses green or red dot + label.
 */
function drawStatsChip(ctx, x, y, face, isSpoof) {
  const emotionLabel = face.emotion?.label;
  const emotionConf  = face.emotion?.confidence;
  const liveLabel    = face.anti_spoofing?.label;
  const liveConf     = face.anti_spoofing?.confidence;

  if (!emotionLabel && !liveLabel) return;

  ctx.font = '500 10px "IBM Plex Mono", monospace';
  const padX = 8;
  const height = 20;

  // Build the parts we'll render with their colors
  const parts = [];
  if (emotionLabel) {
    const text = emotionConf != null
      ? `${capitalize(emotionLabel)} ${(emotionConf * 100).toFixed(0)}%`
      : capitalize(emotionLabel);
    parts.push({ text, color: PAPER_DIM });
  }
  if (liveLabel === 'real') {
    parts.push({
      text: liveConf != null ? `LIVE ${(liveConf * 100).toFixed(0)}%` : 'LIVE',
      color: SIGNAL_GO,
      dot: SIGNAL_GO,
    });
  } else if (liveLabel === 'spoof') {
    parts.push({
      text: liveConf != null ? `SPOOF ${(liveConf * 100).toFixed(0)}%` : 'SPOOF',
      color: SIGNAL_STOP,
      dot: SIGNAL_STOP,
    });
  }

  if (parts.length === 0) return;

  // Measure total width
  const sep = '  ·  ';
  const sepW = ctx.measureText(sep).width;
  const dotW = 12; // 6px dot + 6px gap
  let totalW = 0;
  for (let i = 0; i < parts.length; i++) {
    if (i > 0) totalW += sepW;
    if (parts[i].dot) totalW += dotW;
    totalW += ctx.measureText(parts[i].text).width;
  }

  // Plate background
  ctx.save();
  ctx.shadowColor = 'rgba(0, 0, 0, 0.6)';
  ctx.shadowBlur = 5;
  ctx.shadowOffsetY = 1;
  ctx.fillStyle = BG_PLATE;
  ctx.fillRect(x, y, totalW + padX * 2, height);
  ctx.restore();

  // Render each part
  ctx.textBaseline = 'middle';
  let cx = x + padX;
  const cy = y + height / 2;

  for (let i = 0; i < parts.length; i++) {
    if (i > 0) {
      ctx.fillStyle = 'rgba(232, 223, 201, 0.35)';
      ctx.fillText(sep, cx, cy + 1);
      cx += sepW;
    }
    if (parts[i].dot) {
      ctx.beginPath();
      ctx.fillStyle = parts[i].dot;
      ctx.arc(cx + 3, cy, 3, 0, Math.PI * 2);
      ctx.fill();
      cx += dotW;
    }
    ctx.fillStyle = parts[i].color;
    ctx.fillText(parts[i].text, cx, cy + 1);
    cx += ctx.measureText(parts[i].text).width;
  }
  ctx.textBaseline = 'alphabetic';
}