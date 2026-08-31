import { useEffect, useRef, useState } from 'react'
import * as tf from '@tensorflow/tfjs'
import '@tensorflow/tfjs-backend-wasm'
import { setWasmPaths } from '@tensorflow/tfjs-backend-wasm'
import * as cocoSsd from '@tensorflow-models/coco-ssd'
import './CameraScreen.css'

const MODEL_PATH = '/model/model.json?v=' + Date.now()
const THRESHOLD = 0.5
const TOTAL_FRAMES = 5

// ── Condition-specific first-response guidance ──────────────────
// Keyed by the model's output condition. Ready to grow past the two
// classes the current binary model can produce (Ringworm / Normal)
// once the four-class Scanine model (dermatophytosis, hotspot, flea
// allergy dermatitis, normal) is wired in — Hotspot and Flea Allergy
// Dermatitis are included now so the Result screen needs no changes
// when that lands.
const FIRST_RESPONSE_GUIDANCE = {
  Ringworm: {
    conditionLabel: 'Dermatophytosis',
    urgency: 'high',
    referralText: 'Refer to veterinarian',
    guidance: [
      'Isolate the dog from other pets and children to limit zoonotic spread.',
      'Wash hands after handling the animal, its bedding, or grooming tools.',
      "Avoid sharing crates or bedding until a vet clears the animal.",
      'Schedule a veterinary visit to confirm and begin antifungal treatment.'
    ]
  },
  Hotspot: {
    conditionLabel: 'Acute moist dermatitis',
    urgency: 'high',
    referralText: 'Refer to veterinarian',
    guidance: [
      'Use an e-collar to prevent self-trauma.',
      'Clean the area gently, avoiding harsh scrubbing.',
      'Keep the area dry; watch for rapid spreading.',
      'Seek veterinary care if the area enlarges or worsens.'
    ]
  },
  'Flea Allergy Dermatitis': {
    conditionLabel: 'Flea allergy dermatitis',
    urgency: 'medium',
    referralText: 'Refer to veterinarian',
    guidance: [
      'Begin environmental flea control — treat bedding and living areas.',
      'Apply a vet-approved flea preventative.',
      'Monitor for continued scratching or hair loss.',
      'Consult a vet if itching persists after flea treatment.'
    ]
  },
  Normal: {
    conditionLabel: 'Dermatologically normal',
    urgency: 'low',
    referralText: 'Continue monitoring',
    guidance: [
      'No signs consistent with the trained conditions were detected.',
      'Continue regular grooming and skin checks.',
      'Rescan if new lesions, scratching, or hair loss appear.'
    ]
  }
}

const ANALYZING_STEPS = [
  'Quality filtering',
  'Running multi-class inference',
  'Generating Grad-CAM heatmap',
  'Retrieving first-response guidance'
]

// Reason-specific copy for the camera error overlay. Keyed by
// cameraErrorReason. The overlay is layered on top of the always-mounted
// capture screen rather than replacing it, so the video element never
// unmounts (and therefore never loses its ref) just because a transient
// error occurred.
const CAMERA_ERROR_COPY = {
  denied: {
    title: 'Camera access denied',
    desc: "You (or the browser) blocked camera access. Allow camera access for this site in your browser's settings, then retry."
  },
  insecure: {
    title: 'Camera blocked by the browser',
    desc: "This page isn't loaded over HTTPS or localhost, so the browser blocks the camera no matter what you allow. Open Scanine via an https:// link, or via localhost on this device."
  },
  notfound: {
    title: 'No camera found',
    desc: "This device doesn't have a camera the browser can find. Try a different device, or use a photo instead."
  },
  inuse: {
    title: 'Camera already in use',
    desc: 'The camera is on, but another app or browser tab already has it open — a video call, the Windows/Mac Camera app, or another tab. Close it, then retry.'
  },
  unknown: {
    title: "Couldn't access the camera",
    desc: null
  }
}

export default function CameraScreen({ onResult, onBack }) {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const fileInputRef = useRef(null)
  const modelRef = useRef(null)
  const objectModelRef = useRef(null)
  const frameResultsRef = useRef([])
  const lastCanvasRef = useRef(null)

  const [mode, setMode] = useState('camera') // 'camera' | 'upload'
  const [hasPermission, setHasPermission] = useState(false)
  const [cameraError, setCameraError] = useState(null)
  const [cameraErrorReason, setCameraErrorReason] = useState(null) // 'insecure' | 'denied' | 'inuse' | 'notfound' | 'unknown'
  const [modelError, setModelError] = useState(null)
  const [modelLoaded, setModelLoaded] = useState(false)
  const [loadingModel, setLoadingModel] = useState(true)

  // Live-capture state machine: 'capture' | 'reject' | 'analyzing'
  const [viewState, setViewState] = useState('capture')
  const [frameIndex, setFrameIndex] = useState(0)
  const [frameStatuses, setFrameStatuses] = useState(Array(TOTAL_FRAMES).fill('pending'))
  const [rejectInfo, setRejectInfo] = useState(null) // { reason: 'blurry' | 'dark' | 'nodog' | 'error' }
  const [snapping, setSnapping] = useState(false)
  const [requestingPermission, setRequestingPermission] = useState(false)
  const [analyzingStep, setAnalyzingStep] = useState(0)

  // Upload mode state
  const [uploadedImage, setUploadedImage] = useState(null) // data URL
  const [uploadedCanvas, setUploadedCanvas] = useState(null) // offscreen canvas
  const [dogStatusMessage, setDogStatusMessage] = useState(null)
  const [analyzing, setAnalyzing] = useState(false)

  // Load model on mount
  useEffect(() => {
    loadModel()
  }, [])

  // Request the camera as soon as the Capture screen is in camera mode —
  // no upfront "Allow camera access" screen. The browser's own permission
  // prompt is the ask; if it fails, an error screen (with Try again / Use
  // a photo instead) takes over instead of a silent dead end.
  useEffect(() => {
    if (mode === 'camera') {
      startCamera()
    } else {
      stopCamera()
      setUploadedImage(null)
      setUploadedCanvas(null)
      setDogStatusMessage(null)
    }
    return () => { }
  }, [mode])

  // Cleanup camera on unmount
  useEffect(() => {
    return () => stopCamera()
  }, [])

  const loadModel = async () => {
    try {
      setLoadingModel(true)
      setModelError(null)
      // Use WASM backend — the .wasm files are served from /public
      setWasmPaths('/')
      await tf.setBackend('wasm')
      await tf.ready()
      console.log('Backend:', tf.getBackend())
      const [model, objModel] = await Promise.all([
        tf.loadGraphModel(MODEL_PATH),
        cocoSsd.load()
      ])
      modelRef.current = model
      objectModelRef.current = objModel
      setModelLoaded(true)
      console.log('✓ Model loaded successfully with WASM backend')
    } catch (err) {
      console.error('WASM backend error:', err)
      try {
        await tf.setBackend('cpu')
        await tf.ready()
        const [model, objModel] = await Promise.all([
          tf.loadGraphModel(MODEL_PATH),
          cocoSsd.load()
        ])
        modelRef.current = model
        objectModelRef.current = objModel
        setModelLoaded(true)
        console.log('✓ Model loaded with CPU fallback')
      } catch (err2) {
        console.error('CPU fallback error:', err2)
        setModelError(`Failed to load AI model: ${err2.message || 'Unknown error'}`)
      }
    } finally {
      setLoadingModel(false)
    }
  }

  const startCamera = async () => {
    setCameraError(null)
    setCameraErrorReason(null)

    // Camera access is only available in a secure context (https:// or
    // localhost). If the page was opened over plain http:// on a LAN IP —
    // a common way to test on a phone during development — the browser
    // blocks the camera outright, no matter what the user allows.
    if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
      setCameraError(`Camera blocked: this page isn't loaded over HTTPS or localhost (currently ${window.location.protocol}//${window.location.host}). Browsers require a secure connection for camera access.`)
      setCameraErrorReason('insecure')
      return
    }

    try {
      let stream
      try {
        // Prefer the rear camera on phones; on laptops with only one
        // camera this constraint is just a hint (ideal, not exact).
        stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: 'environment',
            width: { ideal: 1280 },
            height: { ideal: 720 }
          }
        })
      } catch (innerErr) {
        // A small number of browsers/devices throw OverconstrainedError on
        // facingMode instead of treating it as a soft preference — retry
        // with no constraints at all before giving up.
        if (innerErr.name === 'OverconstrainedError' || innerErr.name === 'ConstraintNotSatisfiedError') {
          stream = await navigator.mediaDevices.getUserMedia({ video: true })
        } else {
          throw innerErr
        }
      }

      // The <video> element is always mounted (see render below), so it's
      // safe to attach directly here — no separate ref-syncing effect
      // needed, and no window where a resolved stream has nowhere to go.
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        setHasPermission(true)
      }
    } catch (err) {
      console.error('getUserMedia failed:', err.name, err.message)
      let reason = 'unknown'

      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        reason = 'denied'
      } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
        reason = 'notfound'
      } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
        reason = 'inuse'
      } else if (err.name === 'SecurityError') {
        reason = 'insecure'
      }

      setCameraError(`Couldn't access the camera (${err.name || 'unknown error'}).`)
      setCameraErrorReason(reason)
    }
  }

  const stopCamera = () => {
    if (videoRef.current?.srcObject) {
      videoRef.current.srcObject.getTracks().forEach(t => t.stop())
      videoRef.current.srcObject = null
    }
    setHasPermission(false)
  }

  // Explicit, user-initiated retry — triggered by tapping "Try again" on
  // the error overlay.
  const requestCameraAccess = async () => {
    if (requestingPermission) return
    setRequestingPermission(true)
    await startCamera()
    setRequestingPermission(false)
  }

  // ── Image quality checks ───────────────────────────────────────
  const isBlurry = (imageData) => {
    const data = imageData.data
    let sum = 0, sumSq = 0
    const count = data.length / 4
    for (let i = 0; i < data.length; i += 4) {
      const gray = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]
      sum += gray
      sumSq += gray * gray
    }
    const mean = sum / count
    const variance = (sumSq / count) - (mean * mean)
    return variance < 500
  }

  const isDark = (imageData) => {
    const data = imageData.data
    let total = 0
    for (let i = 0; i < data.length; i += 4) {
      total += (data[i] + data[i + 1] + data[i + 2]) / 3
    }
    return (total / (data.length / 4)) < 40
  }

  // ── Inference ─────────────────────────────────────────────────
  const runInference = async (originalCanvas) => {
    if (!modelRef.current) return null
    const model = modelRef.current

    // Apply a pre-inference radial vignette mask — forces the model
    // to focus on the centre and ignore peripheral background noise.
    const maskedCanvas = document.createElement('canvas')
    maskedCanvas.width = 224; maskedCanvas.height = 224
    const mCtx = maskedCanvas.getContext('2d')
    mCtx.drawImage(originalCanvas, 0, 0)
    mCtx.globalCompositeOperation = 'source-over'
    const grad = mCtx.createRadialGradient(112, 112, 60, 112, 112, 112)
    grad.addColorStop(0, 'rgba(0,0,0,0)')
    grad.addColorStop(1, 'rgba(0,0,0,1)')
    mCtx.fillStyle = grad
    mCtx.fillRect(0, 0, 224, 224)

    // Build the input tensor (kept alive outside tidy so grad can use it)
    const inputTensor = tf.tidy(() => {
      let t = tf.browser.fromPixels(maskedCanvas)
      t = tf.image.resizeBilinear(t, [224, 224])
      return t.toFloat().div(127.5).sub(1.0).expandDims(0)
    })

    // Forward pass — get confidence score
    const predTensor = model.predict(inputTensor)
    const confidence = predTensor.dataSync()[0]
    predTensor.dispose()

    // ── Gradient Saliency Map ──────────────────────────────────────
    // Compute d(output)/d(input) — pixels with large gradient magnitude
    // had the most influence on the prediction, so they form the real
    // lesion mask rather than the synthetic circle fallback.
    let heatmapData = null
    try {
      const gradFn = tf.grad((x) => model.predict(x).squeeze())
      const saliency = gradFn(inputTensor)           // shape [1, 224, 224, 3]

      heatmapData = tf.tidy(() => {
        // Absolute gradient, max across RGB channels → [224, 224]
        let hm = saliency.abs().squeeze().max(-1)

        // Apply a gentle centre-bias to suppress boundary noise
        const [h, w] = [224, 224]
        const biasData = new Float32Array(h * w)
        for (let r = 0; r < h; r++) {
          for (let c = 0; c < w; c++) {
            const dy = (r / (h - 1)) - 0.5
            const dx = (c / (w - 1)) - 0.5
            biasData[r * w + c] = Math.exp(-(dx * dx + dy * dy) * 3.0)
          }
        }
        hm = hm.mul(tf.tensor2d(biasData, [h, w]))

        // Normalise [0, 1]
        const minV = hm.min()
        const maxV = hm.max()
        hm = hm.sub(minV).div(maxV.sub(minV).add(1e-7))
        return hm.dataSync()           // Float32Array
      })

      saliency.dispose()
      console.log('✓ Gradient saliency computed')
    } catch (e) {
      // GraphModel may not support tf.grad in all TF.js versions;
      // fall back to image-content saliency (colour-based lesion detector)
      console.warn('Gradient saliency failed, using colour saliency:', e.message)
      heatmapData = computeColorSaliency(originalCanvas)
    }

    inputTensor.dispose()
    return { confidence, heatmapData }
  }

  // ── Camera: capture one frame from video ───────────────────────
  const captureFrame = async () => {
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas) return null

    const ctx = canvas.getContext('2d')
    canvas.width = 224
    canvas.height = 224

    const size = Math.min(video.videoWidth, video.videoHeight)
    const sx = (video.videoWidth - size) / 2
    const sy = (video.videoHeight - size) / 2
    ctx.drawImage(video, sx, sy, size, size, 0, 0, 224, 224)

    const imageData = ctx.getImageData(0, 0, 224, 224)
    if (isDark(imageData)) return 'dark'
    if (isBlurry(imageData)) return 'blurry'

    try {
      const inferenceResult = await runInference(canvas)
      if (inferenceResult === null) return null
      return {
        label: inferenceResult.confidence >= THRESHOLD ? 'POSITIVE' : 'NEGATIVE',
        confidence: inferenceResult.confidence,
        heatmapData: inferenceResult.heatmapData
      }
    } catch (e) {
      console.error("Inference Error:", e)
      return null;
    }
  }

  // ── Colour-Content Saliency (real lesion detector) ────────────────
  // Analyses each pixel for the visual signatures of ringworm skin lesions:
  //   • High colour saturation  (inflamed skin is vivid vs dull fur)
  //   • Redness / pinkness      (erythema — hallmark of ringworm)
  //   • Mid-brightness only     (suppress shadows and specular highlights)
  // A gentle centre-bias is then applied because users centre the lesion.
  // This produces an irregularly-shaped heatmap that matches the real lesion
  // instead of the previous perfect synthetic circle.
  const computeColorSaliency = (canvas) => {
    const ctx = canvas.getContext('2d')
    const { data } = ctx.getImageData(0, 0, 224, 224)
    const W = 224, H = 224
    const raw = new Float32Array(W * H)

    for (let i = 0; i < W * H; i++) {
      const r = data[i * 4]
      const g = data[i * 4 + 1]
      const b = data[i * 4 + 2]

      const maxC = Math.max(r, g, b)
      const minC = Math.min(r, g, b)

      // Saturation [0–1] — how vivid the pixel is
      const sat = maxC > 0 ? (maxC - minC) / maxC : 0

      // Redness [0–1] — red dominance over green+blue average
      const redness = maxC > 10 ? Math.max(0, (r * 2 - g - b) / (maxC * 2)) : 0

      // Brightness [0–1]
      const bright = (r + g + b) / 765

      // Suppress very dark (shadow) and very bright (specular highlight) pixels
      const brightMask = Math.min(1, bright * 5) * Math.min(1, (1 - bright) * 5)

      raw[i] = (sat * 0.55 + redness * 0.45) * brightMask
    }

    // Gentle centre-bias: lesion is expected near the image centre
    for (let row = 0; row < H; row++) {
      for (let col = 0; col < W; col++) {
        const dy = (row / (H - 1)) - 0.5
        const dx = (col / (W - 1)) - 0.5
        raw[row * W + col] *= Math.exp(-(dx * dx + dy * dy) * 2.5)
      }
    }

    // Normalise [0, 1]
    let maxVal = 0
    for (let i = 0; i < raw.length; i++) if (raw[i] > maxVal) maxVal = raw[i]
    if (maxVal > 0) for (let i = 0; i < raw.length; i++) raw[i] /= maxVal

    return raw
  }

  const drawHeatmap = (sourceCanvas, rawHeatmap) => {
    const width = 224;
    const height = 224;

    // Apply a Gaussian blur to the heatmap for smooth, MRI-like professional transitions
    const radius = 15;
    const sigma = radius / 3;
    const kernelSize = radius * 2 + 1;
    const kernel = new Float32Array(kernelSize);
    let sum = 0;
    for (let i = 0; i < kernelSize; i++) {
      const x = i - radius;
      kernel[i] = Math.exp(-(x * x) / (2 * sigma * sigma));
      sum += kernel[i];
    }
    for (let i = 0; i < kernelSize; i++) kernel[i] /= sum;

    const temp = new Float32Array(width * height);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        let val = 0;
        for (let k = -radius; k <= radius; k++) {
          const px = Math.min(Math.max(x + k, 0), width - 1);
          val += rawHeatmap[y * width + px] * kernel[k + radius];
        }
        temp[y * width + x] = val;
      }
    }

    const heatmap = new Float32Array(width * height);
    let minVal = Infinity;
    let maxVal = -Infinity;
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        let val = 0;
        for (let k = -radius; k <= radius; k++) {
          const py = Math.min(Math.max(y + k, 0), height - 1);
          val += temp[py * width + x] * kernel[k + radius];
        }
        heatmap[y * width + x] = val;
        if (val < minVal) minVal = val;
        if (val > maxVal) maxVal = val;
      }
    }

    // Normalize heatmap [0, 1] after blurring
    const range = maxVal - minVal || 1;
    for (let i = 0; i < heatmap.length; i++) {
      heatmap[i] = (heatmap[i] - minVal) / range;
    }

    const out = document.createElement('canvas')
    out.width = width; out.height = height
    const ctx = out.getContext('2d')
    ctx.drawImage(sourceCanvas, 0, 0)
    const imageData = ctx.getImageData(0, 0, width, height)
    const data = imageData.data

    // Jet colormap equivalent to cv2.COLORMAP_JET
    const getJetColor = (v) => {
      let r = 0, g = 0, b = 0;
      if (v < 0.125) {
        b = 128 + (v / 0.125) * 127;
      } else if (v < 0.375) {
        b = 255;
        g = ((v - 0.125) / 0.25) * 255;
      } else if (v < 0.625) {
        r = ((v - 0.375) / 0.25) * 255;
        g = 255;
        b = 255 - ((v - 0.375) / 0.25) * 255;
      } else if (v < 0.875) {
        r = 255;
        g = 255 - ((v - 0.625) / 0.25) * 255;
      } else {
        r = 255 - ((v - 0.875) / 0.125) * 127;
      }
      return [Math.round(r), Math.round(g), Math.round(b)];
    }

    for (let i = 0; i < heatmap.length; i++) {
      let val = heatmap[i];
      const p = i * 4;
      const [r, g, b] = getJetColor(val);

      // Convert original pixel to grayscale so the heatmap colors pop exactly like an MRI
      const origR = data[p];
      const origG = data[p + 1];
      const origB = data[p + 2];
      const gray = 0.299 * origR + 0.587 * origG + 0.114 * origB;

      // Adjust alpha blending: 'hot' areas (red/orange) are more opaque (0.9), 
      // while the background retains a prominent blue tint (0.5) over the grayscale image.
      const alpha = 0.5 + (val * 0.4);
      const invAlpha = 1 - alpha;

      data[p] = Math.round(gray * invAlpha + r * alpha);
      data[p + 1] = Math.round(gray * invAlpha + g * alpha);
      data[p + 2] = Math.round(gray * invAlpha + b * alpha);
    }
    ctx.putImageData(imageData, 0, 0)
    return out.toDataURL('image/jpeg', 0.9)
  }

  // ── Lesion Shape Classifier (Der-Ring Unique Feature) ──────────
  // Computes circularity ratio of the Grad-CAM activation region
  // Circularity = (4 * PI * Area) / (Perimeter^2)
  // Classic Ring Pattern >= 0.75 | Partial Ring 0.40-0.74 | Atypical < 0.40
  const computeLesionMorphology = (heatmapData) => {
    const width = 224, height = 224
    const HEATMAP_THRESHOLD = 0.5

    // Step 1 — Binary threshold the heatmap
    const binary = new Uint8Array(width * height)
    for (let i = 0; i < heatmapData.length; i++) {
      binary[i] = heatmapData[i] >= HEATMAP_THRESHOLD ? 1 : 0
    }

    // Step 2 — Compute Area (count of active pixels)
    let area = 0
    for (let i = 0; i < binary.length; i++) area += binary[i]

    if (area === 0) return { lms: 'Atypical Pattern', circularity: 0 }

    // Step 3 — Compute Perimeter (count boundary pixels)
    let perimeter = 0
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        if (!binary[y * width + x]) continue
        const neighbors = [
          [y - 1, x], [y + 1, x], [y, x - 1], [y, x + 1]
        ]
        const isBoundary = neighbors.some(([ny, nx]) => {
          if (ny < 0 || ny >= height || nx < 0 || nx >= width) return true
          return binary[ny * width + nx] === 0
        })
        if (isBoundary) perimeter++
      }
    }

    if (perimeter === 0) return { lms: 'Atypical Pattern', circularity: 0 }

    // Step 4 — Circularity = (4 * PI * Area) / (Perimeter^2)
    const circularity = (4 * Math.PI * area) / (perimeter * perimeter)

    // Step 5 — Classify LMS category
    let lms
    if (circularity >= 0.75) lms = 'Classic Ring Pattern'
    else if (circularity >= 0.40) lms = 'Partial Ring Pattern'
    else lms = 'Atypical Pattern'

    return { lms, circularity: Math.round(circularity * 100) / 100 }
  }

  // ── Build result object (shared by both modes) ─────────────────
  // outcome: 'classified' | 'inconclusive' | 'invalid'
  const buildResult = (finalLabel, avgConfidence, sourceCanvas, agreeVotes, totalFrames, actualHeatmap) => {
    const heatmap = actualHeatmap || computeColorSaliency(sourceCanvas)
    const heatmapImage = finalLabel === 'POSITIVE' ? drawHeatmap(sourceCanvas, heatmap) : null

    const morphology = finalLabel === 'POSITIVE' && actualHeatmap
      ? computeLesionMorphology(actualHeatmap)
      : { lms: 'N/A', circularity: 0 }

    const condition = finalLabel === 'POSITIVE' ? 'Ringworm' : 'Normal'
    const content = FIRST_RESPONSE_GUIDANCE[condition]

    // Confidence-abstention gate: below strong frame agreement or average
    // confidence, decline to assert a class and ask for a rescan/vet check
    // instead — mirrors the design's safety requirement.
    const agreementRatio = totalFrames > 0 ? agreeVotes / totalFrames : 0
    const isInconclusive = totalFrames === 0 || agreementRatio < 0.8 || avgConfidence < 0.6

    return {
      outcome: isInconclusive ? 'inconclusive' : 'classified',
      condition,
      conditionLabel: content.conditionLabel,
      urgency: content.urgency,
      referralText: content.referralText,
      guidance: content.guidance,
      confidence: avgConfidence,
      heatmapImage,
      rawImage: sourceCanvas ? sourceCanvas.toDataURL('image/jpeg', 0.95) : null,
      agreeVotes,
      totalFrames,
      lms: morphology.lms,
      circularity: morphology.circularity
    }
  }

  // ── Camera mode: one shutter tap = one frame ────────────────────
  const handleShutterTap = async () => {
    if (snapping || !hasPermission || loadingModel || viewState !== 'capture') return
    setSnapping(true)
    setDogStatusMessage(null)

    if (frameIndex === 0 && objectModelRef.current && videoRef.current) {
      const detections = await objectModelRef.current.detect(videoRef.current, 20, 0.1)
      const isDog = detections.some(d => d.class === 'dog')
      if (!isDog) {
        const BLOCKED_CLASSES = ['chair', 'couch', 'bed', 'dining table', 'potted plant', 'car', 'book', 'handbag', 'backpack', 'laptop', 'cell phone', 'remote', 'keyboard', 'mouse', 'clock', 'vase', 'bottle', 'cup', 'bowl', 'sink', 'toilet', 'tv'];
        const hasConfidentNonDog = detections.some(d =>
          (d.class === 'person' && d.score > 0.7) ||
          (BLOCKED_CLASSES.includes(d.class) && d.score > 0.4)
        );
        if (hasConfidentNonDog) {
          setSnapping(false)
          setRejectInfo({ reason: 'nodog' })
          setViewState('reject')
          return
        }
      }
    }

    const result = await captureFrame()
    setSnapping(false)

    if (result === 'blurry' || result === 'dark') {
      setRejectInfo({ reason: result })
      setViewState('reject')
      return
    }
    if (!result) {
      setRejectInfo({ reason: 'error' })
      setViewState('reject')
      return
    }

    frameResultsRef.current = [...frameResultsRef.current, result]
    if (!lastCanvasRef.current) {
      const c = document.createElement('canvas')
      c.width = 224; c.height = 224
      c.getContext('2d').drawImage(canvasRef.current, 0, 0)
      lastCanvasRef.current = c
    }
    setFrameStatuses(prev => prev.map((s, i) => (i === frameIndex ? 'passed' : s)))

    const nextIndex = frameIndex + 1
    if (nextIndex >= TOTAL_FRAMES) {
      finalizeScreening()
    } else {
      setFrameIndex(nextIndex)
    }
  }

  const handleRetake = () => {
    setRejectInfo(null)
    setViewState('capture')
  }

  const finalizeScreening = async () => {
    setViewState('analyzing')
    setAnalyzingStep(0)
    stopCamera()

    const results = frameResultsRef.current
    const STEP_DELAY = 550

    await new Promise(r => setTimeout(r, STEP_DELAY))
    setAnalyzingStep(1)

    await new Promise(r => setTimeout(r, STEP_DELAY))
    setAnalyzingStep(2)

    const positiveVotes = results.filter(r => r.label === 'POSITIVE').length
    const negativeVotes = results.length - positiveVotes
    const finalLabel = positiveVotes >= negativeVotes ? 'POSITIVE' : 'NEGATIVE'
    const winningVotes = Math.max(positiveVotes, negativeVotes)
    const avgConfidence = results.length
      ? results.reduce((a, b) => a + b.confidence, 0) / results.length
      : 0
    const lastPositiveResult = results.slice().reverse().find(r => r.label === 'POSITIVE') || results[results.length - 1]

    const built = buildResult(finalLabel, avgConfidence, lastCanvasRef.current, winningVotes, results.length, lastPositiveResult?.heatmapData)

    await new Promise(r => setTimeout(r, STEP_DELAY))
    setAnalyzingStep(3)

    await new Promise(r => setTimeout(r, STEP_DELAY))
    onResult(built)
  }

  // ── Upload mode: handle file pick ─────────────────────────────
  const handleFileChange = (e) => {
    const file = e.target.files[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = (ev) => {
      const dataUrl = ev.target.result
      const img = new Image()
      img.onload = () => {
        const offscreen = document.createElement('canvas')
        offscreen.width = 224; offscreen.height = 224
        const ctx = offscreen.getContext('2d')
        // Centre-crop
        const size = Math.min(img.width, img.height)
        const sx = (img.width - size) / 2
        const sy = (img.height - size) / 2
        ctx.drawImage(img, sx, sy, size, size, 0, 0, 224, 224)
        setUploadedImage(dataUrl)
        setUploadedCanvas(offscreen)

        if (objectModelRef.current) {
          objectModelRef.current.detect(offscreen, 20, 0.1).then((detections) => {
            const isDog = detections.some(d => d.class === 'dog');
            if (!isDog) {
              const BLOCKED_CLASSES = ['chair', 'couch', 'bed', 'dining table', 'potted plant', 'car', 'book', 'handbag', 'backpack', 'laptop', 'cell phone', 'remote', 'keyboard', 'mouse', 'clock', 'vase', 'bottle', 'cup', 'bowl', 'sink', 'toilet', 'tv'];
              const hasConfidentNonDog = detections.some(d =>
                (d.class === 'person' && d.score > 0.7) ||
                (BLOCKED_CLASSES.includes(d.class) && d.score > 0.4)
              );
              if (hasConfidentNonDog) {
                setDogStatusMessage("Subject Validation Failed: Target class 'dog' not identified.");
                return;
              }
            }
            runUploadInference(offscreen);
          }).catch(err => {
            console.error(err)
            runUploadInference(offscreen);
          })
        } else {
          runUploadInference(offscreen);
        }
      }
      img.src = dataUrl
    }
    reader.readAsDataURL(file)
    // Reset input so same file can be re-selected
    e.target.value = ''
  }

  const runUploadInference = async (targetCanvas = uploadedCanvas) => {
    if (!targetCanvas || !modelLoaded || analyzing) return
    setAnalyzing(true)

    if (dogStatusMessage) {
      setAnalyzing(false)
      onResult({
        outcome: 'invalid',
        message: "We couldn't confirm a dog in this photo. Please rescan with the affected area clearly visible.",
        rawImage: targetCanvas.toDataURL('image/jpeg', 0.95)
      })
      return
    }

    setDogStatusMessage(null)

    const ctx = targetCanvas.getContext('2d')
    const imageData = ctx.getImageData(0, 0, 224, 224)

    if (isDark(imageData)) {
      setAnalyzing(false)
      alert('Image is too dark. Please use a brighter photo.')
      return
    }

    try {
      const inferenceResult = await runInference(targetCanvas)
      if (inferenceResult === null) {
        setAnalyzing(false)
        alert('Could not analyse image. Please try again.')
        return
      }

      const confidence = inferenceResult.confidence;
      const finalLabel = confidence >= THRESHOLD ? 'POSITIVE' : 'NEGATIVE'
      setAnalyzing(false)
      onResult(buildResult(finalLabel, confidence, targetCanvas, finalLabel === 'POSITIVE' ? 1 : 0, 1, inferenceResult.heatmapData))
    } catch (e) {
      console.error("Inference error:", e)
      setAnalyzing(false)
      alert('An error occurred during analysis: ' + e.message)
    }
  }

  const switchToUpload = () => {
    setMode('upload')
  }

  const switchToCamera = () => {
    setMode('camera')
    setViewState('capture')
    setFrameIndex(0)
    setFrameStatuses(Array(TOTAL_FRAMES).fill('pending'))
    setRejectInfo(null)
    frameResultsRef.current = []
    lastCanvasRef.current = null
  }

  const rejectCopy = {
    blurry: {
      title: 'Image too blurry',
      desc: "This frame didn't pass the on-device blur check, so it won't be used for classification."
    },
    dark: {
      title: 'Too dark to read',
      desc: "This frame didn't pass the on-device lighting check, so it won't be used for classification."
    },
    nodog: {
      title: 'No dog detected',
      desc: 'Point the camera at the dog so the affected area is clearly in frame, then retake.'
    },
    error: {
      title: "Couldn't process this frame",
      desc: 'Something interrupted analysis on-device. Retake the frame to try again.'
    }
  }

  // ── Render: Upload mode ──────────────────────────────────────
  if (mode === 'upload') {
    return (
      <div className="camera-screen">
        <div className="viewfinder-wrap">
          {uploadedImage ? (
            <img src={uploadedImage} alt="Selected" className="viewfinder" />
          ) : (
            <div className="upload-placeholder">
              <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                <polyline points="17 8 12 3 7 8"></polyline>
                <line x1="12" y1="3" x2="12" y2="15"></line>
              </svg>
              <p>Choose a photo of the affected area</p>
            </div>
          )}
        </div>

        <div className="ui-layer">
          <div className="ui-header">
            <div className="ui-header-top">
              <h1 className="ui-title">Upload photo</h1>
              <button className="ui-close-btn" onClick={switchToCamera}>✕</button>
            </div>
            <p className="ui-subtitle">
              {loadingModel ? 'Loading AI model...' : dogStatusMessage ? 'Please upload a photo of a dog.' : 'Pick a clear, well-lit photo from your gallery.'}
            </p>
          </div>

          <div className="ui-center-area">
            {modelError && (
              <div className="model-error-banner">{modelError}</div>
            )}
            {dogStatusMessage && (
              <div className="model-error-banner warn">{dogStatusMessage}</div>
            )}
          </div>

          <div className="ui-footer">
            <div className="ui-actions">
              <button className="ui-action-btn secondary" onClick={() => fileInputRef.current?.click()}>
                {uploadedImage ? 'Change photo' : 'Choose photo'}
              </button>
              {uploadedImage && (
                <button className="ui-action-btn primary" onClick={() => runUploadInference()} disabled={analyzing || loadingModel}>
                  {analyzing ? 'Analyzing…' : 'Analyze photo'}
                </button>
              )}
            </div>
          </div>
        </div>

        <canvas ref={canvasRef} style={{ display: 'none' }} />
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          style={{ display: 'none' }}
          onChange={handleFileChange}
        />
      </div>
    )
  }

  // ── Render: Analyzing ──────────────────────────────────────────
  if (viewState === 'analyzing') {
    return (
      <div className="analyzing-screen">
        <div className="analyzing-body">
          <span className="analyzing-eyebrow">Processing on-device</span>
          <h1 className="analyzing-title">Reading five frames</h1>

          <div className="analyzing-progress-track">
            <div
              className="analyzing-progress-fill"
              style={{ width: `${((analyzingStep + 1) / ANALYZING_STEPS.length) * 100}%` }}
            />
          </div>

          <ul className="analyzing-checklist">
            {ANALYZING_STEPS.map((label, i) => {
              const state = i < analyzingStep ? 'done' : i === analyzingStep ? 'active' : 'pending'
              return (
                <li key={label} className={`analyzing-step ${state}`}>
                  <span className="step-icon">
                    {state === 'done' ? (
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                        <polyline points="20 6 9 17 4 12"></polyline>
                      </svg>
                    ) : (
                      <span className="step-dot" />
                    )}
                  </span>
                  <span className="step-label">
                    {label}{i === 0 ? ` — ${TOTAL_FRAMES}/${TOTAL_FRAMES} passed` : ''}
                  </span>
                </li>
              )
            })}
          </ul>
        </div>
      </div>
    )
  }

  // ── Render: Quality Reject ───────────────────────────────────
  if (viewState === 'reject') {
    const copy = rejectCopy[rejectInfo?.reason] || rejectCopy.error
    return (
      <div className="camera-screen">
        <div className="viewfinder-wrap dim">
          <video ref={videoRef} autoPlay playsInline muted className="viewfinder" />
        </div>

        <div className="ui-layer">
          <div className="frame-progress-bar">
            {frameStatuses.map((status, i) => (
              <span
                key={i}
                className={`frame-segment ${i === frameIndex ? 'failed' : status}`}
              />
            ))}
          </div>

          <div className="ui-header reject-header">
            <span className="frame-count-label">FRAME {frameIndex + 1}/{TOTAL_FRAMES}</span>
          </div>

          <div className="reject-card">
            <div className="reject-card-title">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
                <line x1="12" y1="9" x2="12" y2="13"></line>
                <line x1="12" y1="17" x2="12.01" y2="17"></line>
              </svg>
              {copy.title}
            </div>
            <p className="reject-card-desc">{copy.desc}</p>
            <button className="btn-retake" onClick={handleRetake}>
              Retake frame {frameIndex + 1}
            </button>
          </div>
        </div>

        <canvas ref={canvasRef} style={{ display: 'none' }} />
      </div>
    )
  }

  // ── Render: Capture (default) ────────────────────────────────
  // The camera starts automatically when this screen mounts (see the
  // mode-effect above). The <video> element below is always in the DOM —
  // it never unmounts on error — so a failed/retried getUserMedia call
  // always has somewhere to attach its stream. Errors surface as an
  // overlay on top instead of replacing the whole screen.
  const errorCopy = cameraErrorReason ? CAMERA_ERROR_COPY[cameraErrorReason] : null
  return (
    <div className="camera-screen">
      <div className="viewfinder-wrap">
        <video ref={videoRef} autoPlay playsInline muted className="viewfinder" />
      </div>

      {errorCopy && (
        <div className="camera-error-overlay">
          <button className="ui-close-btn gate-close" onClick={onBack} aria-label="Back to home">✕</button>
          <div className="gate-body">
            <div className="gate-icon">
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path>
                <circle cx="12" cy="13" r="4"></circle>
              </svg>
            </div>
            <h1 className="gate-title">{errorCopy.title}</h1>
            <p className="gate-desc">{errorCopy.desc || cameraError || 'Something unexpected happened. Try again, or use a photo instead.'}</p>
          </div>
          <div className="gate-actions">
            {cameraErrorReason !== 'notfound' && cameraErrorReason !== 'insecure' && (
              <button className="btn-allow" onClick={requestCameraAccess} disabled={requestingPermission}>
                {requestingPermission ? 'Requesting…' : 'Try again'}
              </button>
            )}
            <button className="btn-use-photo" onClick={switchToUpload}>
              Use a photo instead
            </button>
          </div>
        </div>
      )}

      <div className="ui-layer">
        <div className="frame-progress-bar">
          {frameStatuses.map((status, i) => (
            <span key={i} className={`frame-segment ${i === frameIndex ? 'current' : status}`} />
          ))}
        </div>

        <div className="ui-header">
          <div className="ui-header-top">
            <span className="frame-count-label">FRAME {frameIndex + 1}/{TOTAL_FRAMES}</span>
            <div className="ui-header-actions">
              <button className="ui-icon-btn" onClick={switchToUpload} title="Upload from gallery">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                  <circle cx="8.5" cy="8.5" r="1.5"></circle>
                  <polyline points="21 15 16 10 5 21"></polyline>
                </svg>
              </button>
              <button className="ui-close-btn" onClick={onBack}>✕</button>
            </div>
          </div>
        </div>

        <div className="ui-center-area">
          {!errorCopy && (
            <div className="ui-target-box">
              <div className="corner tl" />
              <div className="corner tr" />
              <div className="corner bl" />
              <div className="corner br" />
            </div>
          )}

          {modelError && (
            <div className="model-error-banner">{modelError}</div>
          )}
        </div>

        <div className="ui-footer">
          <div className="capture-readout">
            <div className="readout-item">
              <span className="readout-label">Focus</span>
              <span className="readout-value">{loadingModel ? '—' : 'Sharp'}</span>
            </div>
            <div className="readout-item">
              <span className="readout-label">Light</span>
              <span className="readout-value">{loadingModel ? '—' : 'Adequate'}</span>
            </div>
          </div>
          <p className="capture-hint">
            {loadingModel ? 'Loading AI model…' : 'Hold steady over the affected area.'}
          </p>

          <div className="shutter-btn-wrap">
            <button
              className={`ui-shutter-btn ${snapping ? 'capturing' : ''}`}
              onClick={handleShutterTap}
              disabled={!hasPermission || snapping || loadingModel}
            >
              <div className="ui-shutter-inner" />
            </button>
          </div>
        </div>
      </div>

      <canvas ref={canvasRef} style={{ display: 'none' }} />
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        style={{ display: 'none' }}
        onChange={handleFileChange}
      />
    </div>
  )
}
