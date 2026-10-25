// ============================================================
// 設定
// ============================================================

const SETTINGS = {
  holdSeconds: 1.0,
  prepSeconds: 1.0,
  correctSeconds: 3.0,
  targetCorrect: 3,

  calibrationSeconds: 0.6,
  minVisibility: 0.45,

  // 肩幅を40cmと仮定した相対的な移動判定。
  // 実際の距離を正確に測定するものではありません。
  assumedShoulderWidthCm: 40.0,
  headMoveCm: 10.0,
  headSmoothing: 0.55,

  mouthHandMarginRatio: 0.20,
  poseHandMarginRatio: 0.40,
  mouthMemorySeconds: 3.0,

  soundVolume: 0.35,

  // 推論は最大約15回/秒。
  inferenceIntervalMs: 1000 / 15,

  // 検出が長時間途切れたときは、保持時間をリセット。
  maxDetectionGapSeconds: 0.35,

  keyRepeatSeconds: 0.45
};

const MEDIAPIPE_VERSION = "0.10.14";

const MODULE_URL =
  `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}/vision_bundle.mjs`;

const WASM_URL =
  `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}/wasm`;

const POSE_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/" +
  "pose_landmarker/pose_landmarker_lite/float16/1/" +
  "pose_landmarker_lite.task";

const HAND_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/" +
  "hand_landmarker/hand_landmarker/float16/1/" +
  "hand_landmarker.task";

const COLORS = [
  "#ff6487",
  "#ffc137",
  "#46bfe9",
  "#6bcd69",
  "#b17ce5",
  "#ff934a",
  "#f183c8"
];

// MediaPipe Poseのランドマーク番号。
const LM = {
  NOSE: 0,

  MOUTH_LEFT: 9,
  MOUTH_RIGHT: 10,

  LEFT_SHOULDER: 11,
  RIGHT_SHOULDER: 12,

  LEFT_WRIST: 15,
  RIGHT_WRIST: 16,

  LEFT_PINKY: 17,
  RIGHT_PINKY: 18,

  LEFT_INDEX: 19,
  RIGHT_INDEX: 20,

  LEFT_THUMB: 21,
  RIGHT_THUMB: 22
};

const QUESTIONS = [
  {
    action: "UP",

    lines: [
      [["りょう", ""], ["手", "て"], ["を", ""]],
      [["あげる", ""]]
    ],

    hints: [
      [["鼻", "はな"], ["より", ""], ["上", "うえ"], ["へ", ""]],
      [["1", ""], ["秒", "びょう"], ["ストップ", ""]]
    ]
  },

  {
    action: "RIGHT",

    lines: [
      [["右", "みぎ"], ["に", ""]],
      [["ゆれる", ""]]
    ],

    hints: [
      [["はじめの いちから", ""]],
      [["あたまを ", ""], ["右", "みぎ"], ["へ", ""]],
      [["1", ""], ["秒", "びょう"], ["ストップ", ""]]
    ]
  },

  {
    action: "LEFT",

    lines: [
      [["左", "ひだり"], ["に", ""]],
      [["ゆれる", ""]]
    ],

    hints: [
      [["はじめの いちから", ""]],
      [["あたまを ", ""], ["左", "ひだり"], ["へ", ""]],
      [["1", ""], ["秒", "びょう"], ["ストップ", ""]]
    ]
  },

  {
    action: "COVER_MOUTH",

    lines: [
      [["くちを", ""]],
      [["かくす", ""]]
    ],

    hints: [
      [["かたてでも りょうてでも", ""]],
      [["くちの まえへ", ""]],
      [["1", ""], ["秒", "びょう"], ["ストップ", ""]]
    ]
  }
];

// ============================================================
// HTML要素
// ============================================================

const $ = (id) => document.getElementById(id);

const elements = {
  title: $("title-screen"),
  loading: $("loading-screen"),
  game: $("game-screen"),
  finish: $("finish-screen"),

  start: $("start-button"),
  skip: $("skip-button"),
  end: $("end-button"),
  report: $("report-button"),

  loadingMessage: $("loading-message"),
  titleError: $("title-error"),

  score: $("score"),
  question: $("question-card"),
  hints: $("hints"),

  correct: $("correct-display"),
  status: $("status"),

  progressTrack: $("progress-track"),
  progressFill: $("progress-fill"),

  video: $("camera-video"),
  canvas: $("camera-canvas")
};

for (const [name, element] of Object.entries(elements)) {
  if (!element) {
    throw new Error(
      `必要なHTML要素が見つかりません: ${name}`
    );
  }
}

const context = elements.canvas.getContext("2d");

if (!context) {
  throw new Error("カメラ表示用のCanvasを準備できませんでした。");
}

// ============================================================
// ゲーム状態
// ============================================================

let state = "TITLE";

let poseLandmarker = null;
let handLandmarker = null;
let modelsPromise = null;

let cameraStream = null;
let audioContext = null;

let correctCount = 0;
let questionIndex = 0;
let questionStarted = 0;
let correctStarted = 0;

let deck = [];
let previousQuestion = null;

let headReference = null;
let detector = null;

let confetti = [];

let animationFrameId = null;
let lastFrameTime = null;
let lastInferenceTime = -Infinity;
let lastVideoTime = -1;
let lastKeyTime = -Infinity;

// 非同期処理を無効化するための番号。
let sessionId = 0;

// ============================================================
// 共通関数
// ============================================================

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function midpoint(a, b) {
  return {
    x: (a.x + b.x) / 2,
    y: (a.y + b.y) / 2
  };
}

function median(values) {
  if (values.length === 0) {
    return 0;
  }

  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);

  if (sorted.length % 2 === 1) {
    return sorted[middle];
  }

  return (sorted[middle - 1] + sorted[middle]) / 2;
}

function shuffle(items) {
  for (let i = items.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));

    [items[i], items[j]] = [items[j], items[i]];
  }

  return items;
}

function showScreen(nextState) {
  state = nextState;

  elements.title.hidden = nextState !== "TITLE";
  elements.loading.hidden = nextState !== "LOADING";

  elements.game.hidden =
    nextState !== "PLAY" && nextState !== "CORRECT";

  elements.finish.hidden = nextState !== "FINISH";
}

function setProgress(progress) {
  const percentage = Math.round(
    clamp(progress, 0, 1) * 100
  );

  elements.progressFill.style.width = `${percentage}%`;

  elements.progressTrack.setAttribute(
    "aria-valuenow",
    String(percentage)
  );
}

function setStatus(text) {
  if (elements.status.textContent !== text) {
    elements.status.textContent = text;
  }
}

function showError(message) {
  elements.titleError.textContent = message;
  elements.titleError.hidden = false;
}

function clearError() {
  elements.titleError.textContent = "";
  elements.titleError.hidden = true;
}

function renderRubyLine(parts, className) {
  const line = document.createElement("div");
  line.className = className;

  for (const [base, reading] of parts) {
    if (!reading) {
      line.append(document.createTextNode(base));
      continue;
    }

    const ruby = document.createElement("ruby");
    ruby.append(document.createTextNode(base));

    const rt = document.createElement("rt");
    rt.textContent = reading;

    ruby.append(rt);
    line.append(ruby);
  }

  return line;
}

function stopStream(stream) {
  for (const track of stream.getTracks()) {
    track.stop();
  }
}

function getErrorMessage(error) {
  switch (error?.name) {
    case "NotAllowedError":
    case "PermissionDeniedError":
      return (
        "カメラの使用が許可されませんでした。\n" +
        "ブラウザの設定でカメラを許可して、もう一度試してください。"
      );

    case "NotFoundError":
    case "DevicesNotFoundError":
      return "利用できるカメラが見つかりませんでした。";

    case "NotReadableError":
    case "TrackStartError":
      return (
        "カメラを開始できませんでした。\n" +
        "他のカメラ使用アプリを閉じて、もう一度試してください。"
      );

    default:
      return (
        error?.message ||
        "準備中にエラーが発生しました。ページを再読み込みしてください。"
      );
  }
}

// ============================================================
// モデルの読み込み
// ============================================================

async function loadModels() {
  if (poseLandmarker && handLandmarker) {
    return;
  }

  if (modelsPromise) {
    return modelsPromise;
  }

  modelsPromise = (async () => {
    elements.loadingMessage.textContent =
      "ポーズ判定のプログラムを読み込んでいます";

    const {
      FilesetResolver,
      PoseLandmarker,
      HandLandmarker
    } = await import(MODULE_URL);

    const vision = await FilesetResolver.forVisionTasks(
      WASM_URL
    );

    function createPose(delegate) {
      return PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: POSE_MODEL_URL,
          delegate
        },

        runningMode: "VIDEO",
        numPoses: 1,

        minPoseDetectionConfidence: 0.45,
        minPosePresenceConfidence: 0.45,
        minTrackingConfidence: 0.45,

        // 負荷を抑えるため、人物の切り抜きは行わない。
        outputSegmentationMasks: false
      });
    }

    function createHands(delegate) {
      return HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: HAND_MODEL_URL,
          delegate
        },

        runningMode: "VIDEO",
        numHands: 2,

        minHandDetectionConfidence: 0.35,
        minHandPresenceConfidence: 0.35,
        minTrackingConfidence: 0.35
      });
    }

    let newPose = null;
    let newHands = null;

    try {
      elements.loadingMessage.textContent =
        "ポーズ判定のモデルを読み込んでいます";

      try {
        newPose = await createPose("GPU");
      } catch (error) {
        console.warn(
          "PoseのGPU初期化に失敗しました。CPUへ切り替えます。",
          error
        );

        newPose = await createPose("CPU");
      }

      elements.loadingMessage.textContent =
        "手の判定のモデルを読み込んでいます";

      try {
        newHands = await createHands("GPU");
      } catch (error) {
        console.warn(
          "手のGPU初期化に失敗しました。CPUへ切り替えます。",
          error
        );

        newHands = await createHands("CPU");
      }

      poseLandmarker = newPose;
      handLandmarker = newHands;
    } catch (error) {
      newPose?.close();
      newHands?.close();

      throw error;
    }
  })();

  try {
    await modelsPromise;
  } catch (error) {
    modelsPromise = null;
    throw error;
  }
}

// ============================================================
// カメラ
// ============================================================

async function openCamera() {
  if (!window.isSecureContext) {
    throw new Error(
      "カメラを使うにはHTTPSで公開してください。\n" +
      "iPadでHTMLファイルを直接開く方法には対応していません。"
    );
  }

  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error(
      "このブラウザではカメラを利用できません。\n" +
      "iPadのSafariでHTTPSのページを開いてください。"
    );
  }

  elements.loadingMessage.textContent =
    "カメラの使用を許可してください";

  return navigator.mediaDevices.getUserMedia({
    audio: false,

    video: {
      facingMode: "user",

      width: { ideal: 640 },
      height: { ideal: 480 },

      frameRate: {
        ideal: 24,
        max: 30
      }
    }
  });
}

function stopCamera() {
  if (cameraStream) {
    stopStream(cameraStream);
    cameraStream = null;
  }

  elements.video.pause();
  elements.video.srcObject = null;
}

async function waitForVideoReady(currentSession) {
  const started = performance.now();

  while (
    elements.video.readyState < 2 ||
    elements.video.videoWidth === 0 ||
    elements.video.videoHeight === 0
  ) {
    if (currentSession !== sessionId) {
      return false;
    }

    if (performance.now() - started > 12000) {
      throw new Error(
        "カメラ映像の準備ができませんでした。\n" +
        "他のカメラ使用アプリを閉じて、もう一度試してください。"
      );
    }

    await new Promise((resolve) => {
      setTimeout(resolve, 50);
    });
  }

  return currentSession === sessionId;
}

// ============================================================
// 正解音
// ============================================================

function unlockAudio() {
  try {
    if (!audioContext) {
      const AudioContextClass =
        window.AudioContext || window.webkitAudioContext;

      if (!AudioContextClass) {
        return;
      }

      audioContext = new AudioContextClass();
    }

    audioContext.resume().catch((error) => {
      console.warn("音声を開始できませんでした。", error);
    });
  } catch (error) {
    console.warn("音声を準備できませんでした。", error);
  }
}

function playCorrectSound() {
  if (!audioContext || audioContext.state !== "running") {
    return;
  }

  const startTime = audioContext.currentTime;

  function addNote(frequency, offset, duration) {
    const oscillator = audioContext.createOscillator();
    const gain = audioContext.createGain();

    const noteStart = startTime + offset;
    const noteEnd = noteStart + duration;

    oscillator.type = "sine";
    oscillator.frequency.value = frequency;

    gain.gain.setValueAtTime(0, noteStart);

    gain.gain.linearRampToValueAtTime(
      SETTINGS.soundVolume * 0.5,
      noteStart + 0.008
    );

    gain.gain.exponentialRampToValueAtTime(
      0.001,
      noteEnd
    );

    oscillator.connect(gain);
    gain.connect(audioContext.destination);

    oscillator.start(noteStart);
    oscillator.stop(noteEnd + 0.02);

    oscillator.onended = () => {
      oscillator.disconnect();
      gain.disconnect();
    };
  }

  addNote(1046.5, 0, 0.40);
  addNote(783.99, 0.24, 0.58);
}

// ============================================================
// お題
// ============================================================

function nextQuestionIndex() {
  if (deck.length === 0) {
    deck = shuffle(
      QUESTIONS.map((_, index) => index)
    );

    // デッキの切り替わりでも、同じお題が連続しないようにする。
    if (
      previousQuestion !== null &&
      deck.length > 1 &&
      deck[deck.length - 1] === previousQuestion
    ) {
      const last = deck.length - 1;

      [deck[0], deck[last]] = [deck[last], deck[0]];
    }
  }

  const index = deck.pop();
  previousQuestion = index;

  return index;
}

function startQuestion(now) {
  questionIndex = nextQuestionIndex();
  questionStarted = now;

  detector = new Detector(
    QUESTIONS[questionIndex].action
  );

  elements.question.replaceChildren(
    ...QUESTIONS[questionIndex].lines.map((parts) =>
      renderRubyLine(parts, "question-line")
    )
  );

  elements.hints.replaceChildren(
    ...QUESTIONS[questionIndex].hints.map((parts) =>
      renderRubyLine(parts, "hint-line")
    )
  );

  elements.score.textContent =
    `できた！ ${correctCount} / ${SETTINGS.targetCorrect}`;

  elements.hints.hidden = false;
  elements.correct.hidden = true;

  elements.status.hidden = false;
  elements.progressTrack.hidden = false;

  elements.skip.disabled = false;

  confetti = [];

  setProgress(0);

  setStatus(
    headReference?.ready
      ? "じゅんびしてね"
      : "かおと りょうかたを うつして ストップ"
  );

  showScreen("PLAY");
}

function skipQuestion() {
  if (state !== "PLAY") {
    return;
  }

  startQuestion(performance.now() / 1000);
}

function markCorrect(now) {
  if (state !== "PLAY") {
    return;
  }

  correctCount++;
  correctStarted = now;

  elements.score.textContent =
    `できた！ ${correctCount} / ${SETTINGS.targetCorrect}`;

  elements.hints.hidden = true;
  elements.correct.hidden = false;

  elements.status.hidden = true;
  elements.progressTrack.hidden = true;

  elements.skip.disabled = true;

  startConfetti();
  playCorrectSound();

  showScreen("CORRECT");
}

// ============================================================
// 開始位置の記録
// ============================================================

class HeadReference {
  constructor() {
    this.baseline = null;
    this.samples = [];

    this.started = null;
    this.lastSampleTime = null;
  }

  get ready() {
    return this.baseline !== null;
  }

  clearSamples() {
    this.samples = [];
    this.started = null;
    this.lastSampleTime = null;
  }

  update(data, now) {
    if (this.ready) {
      return true;
    }

    if (!data.nose || !data.width) {
      this.clearSamples();
      return false;
    }

    if (
      this.lastSampleTime !== null &&
      now - this.lastSampleTime >
        SETTINGS.maxDetectionGapSeconds
    ) {
      this.clearSamples();
    }

    this.lastSampleTime = now;

    // 鏡表示に合わせて、横位置を反転する。
    const sample = {
      x: 1 - data.nose.x,
      width: data.width
    };

    if (this.samples.length > 0) {
      const first = this.samples[0];

      const tolerance = Math.max(
        0.008,
        first.width * 0.06
      );

      const movedTooMuch =
        Math.abs(sample.x - first.x) > tolerance;

      const widthChangedTooMuch =
        Math.abs(sample.width - first.width) >
        first.width * 0.18;

      if (movedTooMuch || widthChangedTooMuch) {
        this.samples = [];
        this.started = null;
      }
    }

    if (this.started === null) {
      this.started = now;
    }

    this.samples.push(sample);

    if (
      this.samples.length >= 4 &&
      now - this.started >= SETTINGS.calibrationSeconds
    ) {
      this.baseline = {
        x: median(
          this.samples.map((sample) => sample.x)
        ),

        width: median(
          this.samples.map((sample) => sample.width)
        )
      };

      return true;
    }

    return false;
  }
}

// ============================================================
// ランドマークの計測
// ============================================================

function posePoint(
  landmarks,
  index,
  visibility = SETTINGS.minVisibility
) {
  const point = landmarks?.[index];

  if (!point) {
    return null;
  }

  if (
    !Number.isFinite(point.x) ||
    !Number.isFinite(point.y) ||
    (point.visibility ?? 1) < visibility ||
    point.x < 0 ||
    point.x > 1 ||
    point.y < 0 ||
    point.y > 1
  ) {
    return null;
  }

  return {
    x: point.x,
    y: point.y
  };
}

function collectPoseHand(landmarks, indices) {
  return indices
    .map((index) => posePoint(landmarks, index, 0.20))
    .filter(Boolean);
}

function measure(poseResult, handResult) {
  const landmarks =
    poseResult.landmarks?.[0] ?? null;

  const leftShoulder = posePoint(
    landmarks,
    LM.LEFT_SHOULDER
  );

  const rightShoulder = posePoint(
    landmarks,
    LM.RIGHT_SHOULDER
  );

  let center = null;
  let width = null;

  if (leftShoulder && rightShoulder) {
    const measuredWidth = Math.abs(
      leftShoulder.x - rightShoulder.x
    );

    if (measuredWidth >= 0.04) {
      width = measuredWidth;

      center = midpoint(
        leftShoulder,
        rightShoulder
      );
    }
  }

  const mouthLeft = posePoint(
    landmarks,
    LM.MOUTH_LEFT,
    0.20
  );

  const mouthRight = posePoint(
    landmarks,
    LM.MOUTH_RIGHT,
    0.20
  );

  return {
    nose: posePoint(landmarks, LM.NOSE),

    mouth:
      mouthLeft && mouthRight
        ? midpoint(mouthLeft, mouthRight)
        : null,

    center,
    width,

    faceWidth: width ? width * 0.42 : null,

    lw: posePoint(landmarks, LM.LEFT_WRIST),
    rw: posePoint(landmarks, LM.RIGHT_WRIST),

    hands: handResult?.landmarks ?? [],

    poseHands: [
      collectPoseHand(landmarks, [
        LM.LEFT_WRIST,
        LM.LEFT_INDEX,
        LM.LEFT_PINKY,
        LM.LEFT_THUMB
      ]),

      collectPoseHand(landmarks, [
        LM.RIGHT_WRIST,
        LM.RIGHT_INDEX,
        LM.RIGHT_PINKY,
        LM.RIGHT_THUMB
      ])
    ],

    frameWidth: elements.video.videoWidth,
    frameHeight: elements.video.videoHeight
  };
}

// ============================================================
// 口の位置の保存・補完
// ============================================================

class MouthTracker {
  constructor() {
    this.lastSeen = null;
    this.lastMouth = null;
    this.faceWidth = null;

    this.noseOffset = null;
    this.centerOffset = null;
  }

  update(data, now) {
    if (data.faceWidth) {
      this.faceWidth = data.faceWidth;
    }

    if (data.mouth) {
      this.lastSeen = now;
      this.lastMouth = { ...data.mouth };

      if (data.nose && this.faceWidth) {
        this.noseOffset = {
          x:
            (data.mouth.x - data.nose.x) /
            this.faceWidth,

          y:
            (data.mouth.y - data.nose.y) /
            this.faceWidth
        };
      }

      if (data.center && data.width) {
        this.centerOffset = {
          x:
            (data.mouth.x - data.center.x) /
            data.width,

          y:
            (data.mouth.y - data.center.y) /
            data.width
        };
      }

      return {
        mouth: data.mouth,
        faceWidth: this.faceWidth
      };
    }

    if (
      this.lastSeen !== null &&
      now - this.lastSeen <= SETTINGS.mouthMemorySeconds
    ) {
      if (
        data.nose &&
        this.noseOffset &&
        this.faceWidth
      ) {
        return {
          mouth: {
            x:
              data.nose.x +
              this.noseOffset.x * this.faceWidth,

            y:
              data.nose.y +
              this.noseOffset.y * this.faceWidth
          },

          faceWidth: this.faceWidth
        };
      }

      if (
        data.center &&
        this.centerOffset &&
        data.width
      ) {
        return {
          mouth: {
            x:
              data.center.x +
              this.centerOffset.x * data.width,

            y:
              data.center.y +
              this.centerOffset.y * data.width
          },

          faceWidth: this.faceWidth
        };
      }

      if (this.lastMouth) {
        return {
          mouth: this.lastMouth,
          faceWidth: this.faceWidth
        };
      }
    }

    if (data.nose && this.faceWidth) {
      const aspect =
        data.frameWidth / data.frameHeight;

      return {
        mouth: {
          x: data.nose.x,

          y:
            data.nose.y +
            this.faceWidth * 0.32 * aspect
        },

        faceWidth: this.faceWidth
      };
    }

    return {
      mouth: null,
      faceWidth: this.faceWidth
    };
  }
}

// ============================================================
// 手の領域と口の距離
// ============================================================

function cross(o, a, b) {
  return (
    (a.x - o.x) * (b.y - o.y) -
    (a.y - o.y) * (b.x - o.x)
  );
}

function convexHull(points) {
  const sorted = [...points].sort(
    (a, b) => a.x - b.x || a.y - b.y
  );

  if (sorted.length <= 2) {
    return sorted;
  }

  const lower = [];

  for (const point of sorted) {
    while (
      lower.length >= 2 &&
      cross(
        lower[lower.length - 2],
        lower[lower.length - 1],
        point
      ) <= 0
    ) {
      lower.pop();
    }

    lower.push(point);
  }

  const upper = [];

  for (let i = sorted.length - 1; i >= 0; i--) {
    const point = sorted[i];

    while (
      upper.length >= 2 &&
      cross(
        upper[upper.length - 2],
        upper[upper.length - 1],
        point
      ) <= 0
    ) {
      upper.pop();
    }

    upper.push(point);
  }

  lower.pop();
  upper.pop();

  return lower.concat(upper);
}

function pointInPolygon(point, polygon) {
  let inside = false;

  for (
    let i = 0, j = polygon.length - 1;
    i < polygon.length;
    j = i++
  ) {
    const a = polygon[i];
    const b = polygon[j];

    const intersects =
      (a.y > point.y) !== (b.y > point.y) &&
      point.x <
        ((b.x - a.x) * (point.y - a.y)) /
          (b.y - a.y) +
          a.x;

    if (intersects) {
      inside = !inside;
    }
  }

  return inside;
}

function pointSegmentDistance(point, a, b) {
  const dx = b.x - a.x;
  const dy = b.y - a.y;

  const lengthSquared = dx * dx + dy * dy;

  if (lengthSquared === 0) {
    return Math.hypot(
      point.x - a.x,
      point.y - a.y
    );
  }

  const t = clamp(
    (
      (point.x - a.x) * dx +
      (point.y - a.y) * dy
    ) / lengthSquared,
    0,
    1
  );

  return Math.hypot(
    point.x - (a.x + t * dx),
    point.y - (a.y + t * dy)
  );
}

function distanceToHull(point, hull) {
  if (hull.length === 0) {
    return Infinity;
  }

  if (hull.length === 1) {
    return Math.hypot(
      point.x - hull[0].x,
      point.y - hull[0].y
    );
  }

  if (hull.length === 2) {
    return pointSegmentDistance(
      point,
      hull[0],
      hull[1]
    );
  }

  if (pointInPolygon(point, hull)) {
    return 0;
  }

  let result = Infinity;

  for (let i = 0; i < hull.length; i++) {
    result = Math.min(
      result,

      pointSegmentDistance(
        point,
        hull[i],
        hull[(i + 1) % hull.length]
      )
    );
  }

  return result;
}

function handNearMouth(
  points,
  mouth,
  faceWidth,
  marginRatio,
  data
) {
  if (
    !mouth ||
    !faceWidth ||
    !data.frameWidth ||
    !data.frameHeight
  ) {
    return false;
  }

  // xとyで画像サイズが異なるため、
  // 両方を「画像幅を1とした距離」に変換する。
  const yScale =
    data.frameHeight / data.frameWidth;

  const converted = points
    .filter((point) =>
      Number.isFinite(point.x) &&
      Number.isFinite(point.y)
    )
    .map((point) => ({
      x: point.x,
      y: point.y * yScale
    }));

  if (converted.length === 0) {
    return false;
  }

  const mouthPoint = {
    x: mouth.x,
    y: mouth.y * yScale
  };

  const hull = convexHull(converted);

  return (
    distanceToHull(mouthPoint, hull) <=
    faceWidth * marginRatio
  );
}

// ============================================================
// ポーズ判定
// ============================================================

class Detector {
  constructor(action) {
    this.action = action;

    this.holdStarted = null;
    this.lastUpdate = null;

    this.smoothedHeadX = null;
    this.mouthTracker = new MouthTracker();
  }

  resetHold() {
    this.holdStarted = null;
  }

  checkPose(data, now) {
    if (this.action === "UP") {
      if (!data.nose) {
        return {
          matched: false,
          message: "かおを カメラに うつしてね"
        };
      }

      if (!data.lw || !data.rw) {
        return {
          matched: false,
          message: "りょうてを カメラに うつしてね"
        };
      }

      return {
        matched:
          data.lw.y < data.nose.y &&
          data.rw.y < data.nose.y,

        message: "りょうてを 鼻より 上へ"
      };
    }

    if (
      this.action === "RIGHT" ||
      this.action === "LEFT"
    ) {
      if (!data.nose || !headReference?.ready) {
        this.smoothedHeadX = null;

        return {
          matched: false,
          message: "かおを カメラに うつしてね"
        };
      }

      const mirroredX = 1 - data.nose.x;

      if (this.smoothedHeadX === null) {
        this.smoothedHeadX = mirroredX;
      } else {
        this.smoothedHeadX +=
          SETTINGS.headSmoothing *
          (mirroredX - this.smoothedHeadX);
      }

      const baseline = headReference.baseline;

      const threshold =
        baseline.width *
        (
          SETTINGS.headMoveCm /
          SETTINGS.assumedShoulderWidthCm
        );

      const movement =
        this.smoothedHeadX - baseline.x;

      const isRight = this.action === "RIGHT";

      return {
        matched:
          isRight
            ? movement >= threshold
            : movement <= -threshold,

        message:
          isRight
            ? "はじめの いちから 右へ"
            : "はじめの いちから 左へ"
      };
    }

    if (this.action === "COVER_MOUTH") {
      const tracked = this.mouthTracker.update(
        data,
        now
      );

      if (!tracked.mouth || !tracked.faceWidth) {
        return {
          matched: false,
          message: "かおと りょうかたを うつしてね"
        };
      }

      const detectedHandMatches = data.hands.some(
        (hand) =>
          handNearMouth(
            hand,
            tracked.mouth,
            tracked.faceWidth,
            SETTINGS.mouthHandMarginRatio,
            data
          )
      );

      // 手の詳細モデルで検出できない場合に、
      // Poseの手首・指の点でも補助判定する。
      const poseHandMatches = data.poseHands.some(
        (hand) =>
          handNearMouth(
            hand,
            tracked.mouth,
            tracked.faceWidth,
            SETTINGS.poseHandMarginRatio,
            data
          )
      );

      return {
        matched:
          detectedHandMatches || poseHandMatches,

        message: "てを くちの まえへ"
      };
    }

    return {
      matched: false,
      message: "ポーズをしてね"
    };
  }

  update(data, now) {
    if (
      this.lastUpdate !== null &&
      now - this.lastUpdate >
        SETTINGS.maxDetectionGapSeconds
    ) {
      this.resetHold();
      this.smoothedHeadX = null;
    }

    this.lastUpdate = now;

    const result = this.checkPose(data, now);

    if (!result.matched) {
      this.resetHold();

      return {
        correct: false,
        progress: 0,
        message: result.message
      };
    }

    if (this.holdStarted === null) {
      this.holdStarted = now;
    }

    const elapsed = now - this.holdStarted;

    return {
      correct: elapsed >= SETTINGS.holdSeconds,

      progress:
        elapsed / SETTINGS.holdSeconds,

      message: "そのまま ストップ！"
    };
  }
}

// ============================================================
// カメラ映像の描画
// ============================================================

function resizeCanvas() {
  const rect =
    elements.canvas.getBoundingClientRect();

  if (rect.width <= 0 || rect.height <= 0) {
    return;
  }

  // 高解像度端末でも描画負荷を抑える。
  const pixelRatio = Math.min(
    window.devicePixelRatio || 1,
    1.5
  );

  const width = Math.max(
    1,
    Math.round(rect.width * pixelRatio)
  );

  const height = Math.max(
    1,
    Math.round(rect.height * pixelRatio)
  );

  if (
    elements.canvas.width !== width ||
    elements.canvas.height !== height
  ) {
    elements.canvas.width = width;
    elements.canvas.height = height;
  }
}

function drawCamera() {
  resizeCanvas();

  const width = elements.canvas.width;
  const height = elements.canvas.height;

  context.clearRect(0, 0, width, height);

  context.fillStyle = "#e9e2c8";
  context.fillRect(0, 0, width, height);

  const videoWidth = elements.video.videoWidth;
  const videoHeight = elements.video.videoHeight;

  if (
    elements.video.readyState < 2 ||
    !videoWidth ||
    !videoHeight
  ) {
    return;
  }

  // 映像全体を表示する。
  // 縦横比が合わない部分は背景色の余白にする。
  const scale = Math.min(
    width / videoWidth,
    height / videoHeight
  );

  const drawWidth = videoWidth * scale;
  const drawHeight = videoHeight * scale;

  const x = (width - drawWidth) / 2;
  const y = (height - drawHeight) / 2;

  context.save();

  // 鏡のように左右反転する。
  context.translate(width, 0);
  context.scale(-1, 1);

  context.drawImage(
    elements.video,
    x,
    y,
    drawWidth,
    drawHeight
  );

  context.restore();
}

// ============================================================
// 紙吹雪
// ============================================================

function startConfetti() {
  confetti = Array.from(
    { length: 85 },
    () => ({
      x: Math.random(),
      y: -Math.random() * 0.8,

      vx: (Math.random() - 0.5) * 0.20,
      vy: 0.18 + Math.random() * 0.24,

      size: 6 + Math.random() * 8,

      angle: Math.random() * Math.PI * 2,

      rotation:
        (Math.random() - 0.5) * 8,

      color:
        COLORS[
          Math.floor(Math.random() * COLORS.length)
        ]
    })
  );
}

function drawConfetti(deltaSeconds) {
  const width = elements.canvas.width;
  const height = elements.canvas.height;

  const sizeScale = Math.max(
    0.7,
    width / 600
  );

  for (const particle of confetti) {
    particle.x += particle.vx * deltaSeconds;
    particle.y += particle.vy * deltaSeconds;

    particle.angle +=
      particle.rotation * deltaSeconds;

    if (particle.y > 1.12) {
      particle.y = -0.10;
      particle.x = Math.random();
    }

    context.save();

    context.translate(
      particle.x * width,
      particle.y * height
    );

    context.rotate(particle.angle);
    context.fillStyle = particle.color;

    const size = particle.size * sizeScale;

    context.fillRect(
      -size / 2,
      -size / 3,
      size,
      size * 0.65
    );

    context.restore();
  }
}

// ============================================================
// 推論・ゲーム更新
// ============================================================

function processDetection(data, now) {
  if (state !== "PLAY") {
    return;
  }

  if (!headReference.ready) {
    const ready = headReference.update(data, now);

    detector.resetHold();
    setProgress(0);

    if (ready) {
      // 開始位置の記録後に準備時間を取り直す。
      questionStarted = now;
      setStatus("じゅんびできたよ！");
    } else {
      setStatus(
        "かおと りょうかたを うつして ストップ"
      );
    }

    return;
  }

  if (
    now - questionStarted <
    SETTINGS.prepSeconds
  ) {
    // 準備中も、口の位置は保存しておく。
    detector.mouthTracker.update(data, now);

    detector.resetHold();
    setProgress(0);
    setStatus("じゅんびしてね");

    return;
  }

  const result = detector.update(data, now);

  setProgress(result.progress);
  setStatus(result.message);

  if (result.correct) {
    markCorrect(now);
  }
}

function runInference(timestampMs) {
  if (
    state !== "PLAY" ||
    !poseLandmarker ||
    !handLandmarker
  ) {
    return;
  }

  if (elements.video.readyState < 2) {
    detector?.resetHold();
    setProgress(0);
    setStatus("カメラ映像を まっています");

    return;
  }

  if (
    timestampMs - lastInferenceTime <
    SETTINGS.inferenceIntervalMs
  ) {
    return;
  }

  if (
    elements.video.currentTime === lastVideoTime
  ) {
    // カメラ映像が止まったときに、
    // 最後のポーズのまま正解にならないようにする。
    if (
      timestampMs - lastInferenceTime >
      SETTINGS.maxDetectionGapSeconds * 1000
    ) {
      detector?.resetHold();
      setProgress(0);
      setStatus("カメラ映像を まっています");
    }

    return;
  }

  lastInferenceTime = timestampMs;
  lastVideoTime = elements.video.currentTime;

  const poseResult = poseLandmarker.detectForVideo(
    elements.video,
    timestampMs
  );

  let handResult = null;

  // 詳細な手の推論は「くちをかくす」のときだけ行う。
  if (
    QUESTIONS[questionIndex].action ===
      "COVER_MOUTH" &&
    headReference.ready
  ) {
    handResult = handLandmarker.detectForVideo(
      elements.video,
      timestampMs
    );
  }

  const data = measure(
    poseResult,
    handResult
  );

  processDetection(
    data,
    timestampMs / 1000
  );
}

function stopAnimationLoop() {
  if (animationFrameId !== null) {
    cancelAnimationFrame(animationFrameId);
    animationFrameId = null;
  }

  lastFrameTime = null;
}

function animationLoop(timestampMs) {
  animationFrameId = null;

  if (
    state !== "PLAY" &&
    state !== "CORRECT"
  ) {
    return;
  }

  try {
    const deltaSeconds =
      lastFrameTime === null
        ? 0
        : clamp(
            (timestampMs - lastFrameTime) / 1000,
            0,
            0.05
          );

    lastFrameTime = timestampMs;

    drawCamera();

    if (state === "PLAY") {
      runInference(timestampMs);
    }

    if (state === "CORRECT") {
      drawConfetti(deltaSeconds);

      const now = timestampMs / 1000;

      if (
        now - correctStarted >=
        SETTINGS.correctSeconds
      ) {
        if (
          correctCount >=
          SETTINGS.targetCorrect
        ) {
          finishGame();
        } else {
          startQuestion(now);
        }
      }
    }
  } catch (error) {
    console.error("ゲームの実行中にエラーが発生しました。", error);

    returnToTitle();

    showError(
      "ポーズ判定中にエラーが発生しました。\n" +
      getErrorMessage(error)
    );

    return;
  }

  if (
    state === "PLAY" ||
    state === "CORRECT"
  ) {
    animationFrameId =
      requestAnimationFrame(animationLoop);
  }
}

function startAnimationLoop() {
  stopAnimationLoop();

  lastInferenceTime = -Infinity;
  lastVideoTime = -1;

  animationFrameId =
    requestAnimationFrame(animationLoop);
}

// ============================================================
// スタート・終了
// ============================================================

async function startGame() {
  // 二重スタートを防ぐ。
  if (state !== "TITLE") {
    return;
  }

  const currentSession = ++sessionId;

  clearError();
  unlockAudio();

  elements.start.disabled = true;

  correctCount = 0;

  deck = [];
  previousQuestion = null;

  headReference = new HeadReference();
  detector = null;

  confetti = [];

  showScreen("LOADING");

  try {
    // ボタン操作の直後にカメラ許可を要求する。
    const stream = await openCamera();

    if (currentSession !== sessionId) {
      stopStream(stream);
      return;
    }

    cameraStream = stream;

    elements.video.muted = true;
    elements.video.autoplay = true;
    elements.video.playsInline = true;

    elements.video.setAttribute(
      "playsinline",
      ""
    );

    elements.video.srcObject = stream;

    elements.loadingMessage.textContent =
      "カメラ映像を準備しています";

    await elements.video.play();

    if (currentSession !== sessionId) {
      return;
    }

    const videoReady = await waitForVideoReady(
      currentSession
    );

    if (!videoReady) {
      return;
    }

    await loadModels();

    if (currentSession !== sessionId) {
      return;
    }

    const videoTrack =
      cameraStream?.getVideoTracks()[0];

    if (
      !videoTrack ||
      videoTrack.readyState !== "live"
    ) {
      throw new Error(
        "カメラが停止しました。もう一度スタートしてください。"
      );
    }

    // OSなどによってカメラが停止した場合。
    videoTrack.addEventListener(
      "ended",
      () => {
        if (
          currentSession === sessionId &&
          (
            state === "PLAY" ||
            state === "CORRECT"
          )
        ) {
          returnToTitle();

          showError(
            "カメラが停止しました。\n" +
            "もう一度スタートしてください。"
          );
        }
      },
      { once: true }
    );

    startQuestion(
      performance.now() / 1000
    );

    resizeCanvas();
    startAnimationLoop();
  } catch (error) {
    if (currentSession !== sessionId) {
      return;
    }

    console.error("ゲームを開始できませんでした。", error);

    returnToTitle();
    showError(getErrorMessage(error));
  } finally {
    if (currentSession === sessionId) {
      elements.start.disabled = false;
    }
  }
}

function finishGame() {
  // カメラ・開始待ちの非同期処理を無効化する。
  sessionId++;

  stopAnimationLoop();
  stopCamera();

  confetti = [];
  detector = null;

  elements.start.disabled = false;

  showScreen("FINISH");

  elements.report.focus({
    preventScroll: true
  });
}

function returnToTitle() {
  sessionId++;

  stopAnimationLoop();
  stopCamera();

  detector = null;
  headReference = null;

  confetti = [];

  elements.start.disabled = false;
  elements.skip.disabled = false;

  setProgress(0);

  showScreen("TITLE");

  elements.start.focus({
    preventScroll: true
  });
}

// ============================================================
// ボタン・キーボード
// ============================================================

elements.start.addEventListener(
  "click",
  startGame
);

elements.skip.addEventListener(
  "click",
  skipQuestion
);

elements.end.addEventListener(
  "click",
  returnToTitle
);

// このボタンはタイトルに戻る動作。
// 外部への報告送信は行わない。
elements.report.addEventListener(
  "click",
  returnToTitle
);

document.addEventListener(
  "keydown",
  (event) => {
    if (
      event.isComposing ||
      event.ctrlKey ||
      event.metaKey ||
      event.altKey
    ) {
      return;
    }

    const target = event.target;

    if (
      target instanceof HTMLElement &&
      (
        target.isContentEditable ||
        target.matches(
          "input, textarea, select"
        )
      )
    ) {
      return;
    }

    const isD =
      event.code === "KeyD" ||
      event.key?.toLowerCase() === "d";

    if (!isD) {
      return;
    }

    event.preventDefault();

    // 長押しによる連続操作を防ぐ。
    if (event.repeat) {
      return;
    }

    const now = performance.now() / 1000;

    if (
      now - lastKeyTime <
      SETTINGS.keyRepeatSeconds
    ) {
      return;
    }

    lastKeyTime = now;

    if (state === "TITLE") {
      startGame();
    } else if (state === "PLAY") {
      skipQuestion();
    } else if (state === "FINISH") {
      returnToTitle();
    }
  }
);

// ============================================================
// 画面サイズ・ページ離脱
// ============================================================

window.addEventListener(
  "resize",
  () => {
    if (
      state === "PLAY" ||
      state === "CORRECT"
    ) {
      resizeCanvas();
    }
  }
);

window.addEventListener(
  "pagehide",
  () => {
    // ページを離れたときは必ずカメラを停止する。
    returnToTitle();
  }
);

document.addEventListener(
  "visibilitychange",
  () => {
    if (
      document.hidden &&
      state === "PLAY"
    ) {
      // 別のアプリ・タブへ移動した時間は、
      // ポーズの保持時間に含めない。
      detector?.resetHold();

      if (detector) {
        detector.lastUpdate = null;
        detector.smoothedHeadX = null;
      }

      if (
        headReference &&
        !headReference.ready
      ) {
        headReference.clearSamples();
      }

      setProgress(0);
    }

    if (!document.hidden) {
      lastFrameTime = null;
    }
  }
);

// ============================================================
// 初期表示
// ============================================================

clearError();
setProgress(0);

elements.start.disabled = false;

showScreen("TITLE");

console.log("ポーズでクリア：app.jsの読み込み完了");
