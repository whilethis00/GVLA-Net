const fs = require("fs");
const path = require("path");

const PptxGenJS = require("pptxgenjs");
const { PDFDocument, rgb, StandardFonts } = require("pdf-lib");
const fontkit = require("fontkit");

const ROOT = "/home/introai11/.agile/users/hsjung/projects/GVLA-Net";
const FIG = path.join(ROOT, "experiments/results/figures");
const ASSET = path.join(ROOT, "docs/assets");
const OUT_DIR = path.join(ROOT, "docs");
const PPTX_OUT = path.join(OUT_DIR, "GVLA_Professor_Proposal.pptx");
const PDF_OUT = path.join(OUT_DIR, "GVLA_Professor_Proposal.pdf");

const FONT_PATH = path.join(ASSET, "NotoSansCJKkr-Regular.otf");

const COLORS = {
  navy: "0F172A",
  blue: "1D4ED8",
  blue2: "2563EB",
  sky: "DBEAFE",
  ink: "111827",
  gray: "6B7280",
  light: "F8FAFC",
  line: "CBD5E1",
  green: "0F766E",
  amber: "B45309",
  red: "B91C1C",
  white: "FFFFFF",
};

const slides = [
  {
    kind: "title",
    title: "GVLA-Net",
    subtitle: "Geometric Vision-Language-Action Network",
    kicker: "Breaking the Action Inference Wall from O(N) to O(log N)",
    bullets: [
      "action을 전부 비교하지 않고 찾을 수 없을까?",
      "large action space에서 더 가벼운 routing이 가능할까?",
      "GVLA는 이 질문에서 출발",
    ],
  },
  {
    title: "왜 지금 action head를 다시 봐야 하나",
    subtitle: "backbone은 커졌는데, 마지막 action interface는 아직 무겁다",
    bullets: [
      "backbone은 계속 강해짐",
      "그런데 action head는 여전히 후보를 많이 봐야 함",
      "action space가 커질수록 latency / memory 부담도 같이 커짐",
    ],
    notes: [
      "요즘 VLA는 backbone 쪽은 빠르게 좋아지고 있음",
      "하지만 마지막 action selection은 큰 후보 집합을 직접 다루는 경우가 많음",
      "그래서 action space가 커질수록 병목이 생길 수 있음",
    ],
  },
  {
    title: "우리가 건드리고 싶은 지점",
    subtitle: "학습을 더 잘하는 문제보다, 추론을 더 싸게 만드는 문제",
    bullets: [
      "요즘 VLA는 크게 세 방향",
      "더 큰 backbone",
      "더 나은 action representation",
      "더 효율적인 inference / deployment",
      "GVLA는 action interface와 routing cost에 집중",
    ],
    notes: [
      "GVLA는 backbone을 바꾸려는 게 아님",
      "학습이 무조건 더 잘된다는 주장도 아님",
      "추론 시점의 action routing cost를 어떻게 줄일지에 초점",
    ],
  },
  {
    title: "우리가 바꾸려는 것 / 안 바꾸려는 것",
    subtitle: "VLA 전체가 아니라 action inference interface",
    twoColLists: {
      leftTitle: "바꾸려는 것",
      left: [
        "dense action scoring",
        "큰 discrete action dictionary를 직접 비교하는 구조",
        "선형 탐색형 action routing",
      ],
      rightTitle: "안 바꾸려는 것",
      right: [
        "visual encoder",
        "language backbone",
        "planner / memory / world model 전체",
        "데이터셋 자체",
      ],
    },
    notes: [
      "OpenVLA나 Octo 전체를 갈아엎자는 얘기는 아님",
      "backbone이 latent를 만든 뒤 마지막에 action으로 보내는 인터페이스를 다시 설계해보자는 제안",
    ],
  },
  {
    title: "주장 범위와 flow matching 관계",
    subtitle: "learning improvement라기보다 inference improvement",
    bullets: [
      "직접 겨냥: large action regime latency, memory cost, dense enumeration 비효율",
      "직접 주장 안 함: 학습이 항상 더 잘 됨, 모든 continuous policy 대체",
      "dense classification head / token-level routing은 직접 비교 대상",
      "diffusion / flow matching은 완전한 1:1 대체 관계로 보기 어려움",
      "더 정확한 표현: action interface 쪽의 다른 선택지",
    ],
    notes: [
      "현실적으로 주장할 수 있는 범위를 지키는 게 중요",
      "flow matching 전체를 대체한다고 말하면 무리",
      "action을 어떻게 parameterize하고 추론할지에 대한 대안으로 제시",
    ],
  },
  {
    title: "왜 기존 방식이 무거워지는가",
    subtitle: "후보를 전부 보는 구조라서",
    code: [
      "score_i = <s, a_i>,  i = 1, ..., N",
      "action = argmax_i score_i",
    ],
    bullets: [
      "후보 수 N만큼 비교",
      "메모리도 N에 따라 커짐",
      "action space가 커질수록 선형적으로 부담 증가",
    ],
    notes: [
      "softmax 자체가 문제라기보다 모든 후보를 직접 보는 구조가 문제",
      "후보가 작을 때는 괜찮지만 3만 개, 100만 개로 가면 부담이 커짐",
    ],
  },
  {
    title: "양자역학에서 가져온 직관",
    subtitle: "직교한 관측은 중복이 적다",
    bullets: [
      "orthogonal measurement",
      "겹치지 않는 정보",
      "독립적인 질문",
      "action routing에 이 직관을 가져오기",
    ],
    notes: [
      "공식을 그대로 가져온 건 아님",
      "직교한 관측은 서로 중복이 적다는 직관을 가져옴",
      "latent action space에서 각 projection이 다른 질문 역할을 할 수 있다고 봄",
    ],
  },
  {
    title: "핵심 아이디어와 수식",
    subtitle: "다 점수 매기지 말고, 질문으로 좁혀 가기",
    code: [
      "s in R^d",
      "W in R^(k x d),  k = ceil(log2 N)",
      "y = W s",
      "b = sign(y)",
      "L_ortho = ||W W^T - I||_F^2",
    ],
    bullets: [
      "기존: 모든 action 후보를 직접 점수 매김",
      "GVLA: k = ceil(log2 N)개의 binary question",
      "각 질문의 답을 bit로 모아서 action code 생성",
      "질문 방향 W도 같이 학습",
    ],
    notes: [
      "scoring에서 routing으로 바꾸는 것이 핵심",
      "latent를 몇 개의 방향으로 투영하고 부호를 보며 bit를 만듦",
      "이 방향들이 서로 최대한 겹치지 않게 학습됨",
    ],
  },
  {
    title: "왜 직교이고 왜 log N인가",
    subtitle: "bit 수만 많다고 정보가 늘어나는 건 아님",
    code: ["2^k >= N", "k = ceil(log2 N)"],
    bullets: [
      "비슷한 질문 = 중복된 bit",
      "중복된 bit = 낮은 code capacity",
      "orthogonality = 각 bit가 다른 정보 담당",
      "N개를 구분하는 데 필요한 yes/no 질문 수가 log2(N)",
    ],
    notes: [
      "질문끼리 비슷하면 같은 얘기를 반복하게 됨",
      "orthogonality는 예쁜 제약이 아니라 bit를 제대로 쓰기 위한 핵심 조건",
      "log N은 구현 트릭보다 정보량 관점에서 자연스럽게 나옴",
    ],
  },
  {
    title: "실험에서 보여주고 싶은 것",
    subtitle: "속도만이 아니라 구조와 해석까지",
    table: {
      headers: ["실험 축", "보고 싶은 질문"],
      rows: [
        ["Scaling", "action space가 커질수록 진짜 유리한가"],
        ["Transplant", "여러 backbone에서도 통하는가"],
        ["Orthogonality ablation", "직교성이 진짜 핵심인가"],
        ["Entropy / correlation", "내부 메커니즘이 보이는가"],
        ["Tracking", "control 쪽에서도 의미가 있는가"],
      ],
    },
    notes: [
      "실험 하나만 잘 나오면 부족하다고 생각",
      "속도, 구조, 메커니즘, 응용성, 범용성을 따로 나눠서 봄",
    ],
  },
  {
    title: "Experiment 1. Scaling",
    subtitle: "action space가 커질수록 gap이 어떻게 변하나",
    bullets: [
      "배경: GVLA의 핵심 가설은 large action regime에서 더 잘 드러남",
      "세팅: N = 1,024 / 32,768 / 1,048,576",
      "dense head vs GVLA head latency 비교",
    ],
    notes: [
      "절대 시간 자랑이 목적은 아님",
      "N이 커질수록 두 방식의 scaling 모양이 얼마나 다르게 보이느냐가 핵심",
    ],
  },
  {
    title: "Experiment 1. Scaling Result",
    subtitle: "N이 커질수록 GVLA 쪽 advantage가 커짐",
    image: path.join(FIG, "fig_pareto_efficiency.png"),
    table: {
      headers: ["Num Actions", "Dense Head", "GVLA Head", "Speedup"],
      rows: [
        ["1,024", "2.77 ms", "0.136 ms", "20.38x"],
        ["32,768", "13.05 ms", "0.148 ms", "88.03x"],
        ["1,048,576", "342.00 ms", "0.142 ms", "2410.31x"],
      ],
    },
    notes: [
      "중요한 건 숫자 하나보다 추세",
      "action space가 커질수록 GVLA 쪽이 더 유리해지는 모양이 분명히 보임",
    ],
  },
  {
    title: "Experiment 2. Cross-Backbone Transplant",
    subtitle: "한 모델 전용 trick인지 아닌지 보기",
    bullets: [
      "배경: local trick이면 임팩트가 약함",
      "세팅: Octo-Base / OpenVLA-7B / RT-2-X / pi0.5",
      "여러 backbone에서 같은 방향성이 나오는지 확인",
    ],
    notes: ["GVLA가 특정 backbone에서만 통하는 아이디어인지 아닌지를 보기 위한 실험"],
  },
  {
    title: "Experiment 2. Universal Head Transplant",
    subtitle: "large action regime에서 반복적으로 비슷한 방향성이 나옴",
    table: {
      headers: ["Model", "Actions", "Dense ms", "GVLA ms", "Speedup", "Dense MB", "GVLA MB"],
      rows: [
        ["Octo-Base", "1,048,576", "342.00", "0.14", "2410.31x", "3072", "0.059"],
        ["OpenVLA-7B", "1,048,576", "5.21", "0.12", "44.02x", "16384", "0.313"],
        ["RT-2-X", "1,048,576", "354.66", "0.17", "2072.11x", "16384", "0.313"],
        ["pi0.5", "1,048,576", "1.76", "0.16", "10.94x", "2048", "0.039"],
      ],
    },
    bullets: [
      "backbone마다 절대값은 다름",
      "그래도 large N으로 갈수록 advantage가 커지는 패턴은 반복",
      "memory reduction도 꽤 큼",
    ],
    notes: [
      "모든 setting에서 무조건 이긴다고 말할 필요는 없음",
      "큰 action space로 갈수록 강점이 커진다고 말하는 편이 더 현실적",
    ],
  },
  {
    title: "Experiment 3. Orthogonality Ablation",
    subtitle: "직교성이 핵심이면, 깨졌을 때 성능도 같이 무너져야 함",
    bullets: [
      "배경: orthogonality가 진짜 핵심인지 확인 필요",
      "세팅: orthogonal reg. 유무 비교",
      "collision rate / unique code ratio / row correlation 측정",
    ],
    notes: ["orthogonality가 보기 좋아서 들어간 게 아니라는 점을 보여주기 위한 실험"],
  },
  {
    title: "Experiment 3. Orthogonality Result",
    subtitle: "직교성이 깨지면 code quality가 눈에 띄게 나빠짐",
    image: path.join(FIG, "ablation_orthogonality_paper.png"),
    table: {
      headers: ["Bits", "Method", "Collision", "Unique Ratio", "Row Cosine"],
      rows: [
        ["20", "Ours", "0.6314", "0.6325", "0.0000"],
        ["20", "w/o Ortho", "0.8946", "0.2202", "0.2084"],
        ["22", "Ours", "0.2205", "0.8852", "0.0000"],
        ["22", "w/o Ortho", "0.7738", "0.3752", "0.1898"],
        ["24", "Ours", "0.0604", "0.9695", "0.0000"],
        ["24", "w/o Ortho", "0.6303", "0.5173", "0.2113"],
      ],
    },
    notes: [
      "orthogonal reg.가 있으면 collision이 훨씬 낮음",
      "중요한 건 bit 수보다 bit 품질",
    ],
  },
  {
    title: "Experiment 4. Correlation / Overlap",
    subtitle: "orthogonal basis가 더 덜 겹침",
    imagePair: [
      path.join(FIG, "fig1_orthogonality_heatmap_paper.png"),
      path.join(FIG, "fig4_correlation_sweep_paper.png"),
    ],
    bullets: [
      "orthogonal 쪽이 unique code rate가 더 높음",
      "info overlap은 거의 0에 가까움",
      "bit가 서로 다른 역할을 한다는 해석과 잘 맞음",
    ],
    notes: ["직교성이 실제로 중복을 줄여주고 있다는 걸 보여주고 싶었던 슬라이드"],
  },
  {
    title: "Experiment 4. Entropy Waterfall",
    subtitle: "bit가 늘수록 후보군이 빠르게 줄어듦",
    image: path.join(FIG, "fig_entropy_waterfall_measured.png"),
    table: {
      headers: ["Bit", "Candidate Count", "Entropy(bits)", "Peak Prob."],
      rows: [
        ["1", "8,388,608", "22.10", "3.18e-07"],
        ["8", "65,536", "16.00", "1.53e-05"],
        ["16", "256", "8.00", "3.91e-03"],
        ["20", "16", "4.00", "6.25e-02"],
        ["24", "1", "0.00", "1.00"],
      ],
    },
    notes: ["GVLA가 어떻게 후보를 줄여 가는지 가장 직관적으로 보여주는 figure"],
  },
  {
    title: "Experiment 5. Tracking / Control",
    subtitle: "이게 실제 control 쪽에서도 의미가 있을까",
    bullets: [
      "배경: head benchmark만으로는 practical relevance가 약함",
      "세팅: 131,072 / 524,288 / 1,048,576",
      "GVLA controller vs dense controller",
      "latency / FPS / final error 비교",
    ],
    notes: ["속도만 빠르다고 끝이 아니니까 control 관점에서도 한 번 보자는 실험"],
  },
  {
    title: "Experiment 5. Tracking Result",
    subtitle: "action space가 커질수록 control 쪽 차이도 보이기 시작",
    image: path.join(FIG, "fig_tracking_scaling.png"),
    table: {
      headers: ["Action Space", "Ctrl", "Latency", "FPS", "Final Error"],
      rows: [
        ["131,072", "GVLA", "1.492", "670.36", "0.0278"],
        ["131,072", "Dense", "0.893", "1119.22", "0.0377"],
        ["524,288", "GVLA", "1.672", "598.00", "0.0417"],
        ["524,288", "Dense", "3.548", "281.88", "0.0376"],
        ["1,048,576", "GVLA", "1.632", "612.61", "0.0277"],
        ["1,048,576", "Dense", "6.847", "146.06", "0.0376"],
      ],
    },
    notes: [
      "131k에서는 dense가 더 빠르지만 GVLA error가 더 낮음",
      "524k와 1M에서는 GVLA가 latency / FPS에서도 우세",
    ],
  },
  {
    title: "지금까지 결과를 한 문장으로 묶으면",
    subtitle: "large action regime에서 action inference 구조를 다시 볼 필요가 있음",
    bullets: [
      "dense action scoring은 큰 action space에서 무거워짐",
      "GVLA는 질문 기반 routing으로 구조를 바꿔보려 함",
      "scaling 결과는 이 차이가 커질 수 있음을 보여줌",
      "ablation은 orthogonality가 핵심임을 보여줌",
      "tracking / transplant는 응용성과 범용성을 뒷받침",
    ],
    notes: [
      "large action space에서는 action inference 구조 자체를 다시 볼 필요가 있음",
      "GVLA는 그 한 가지 꽤 강한 후보가 될 수 있다는 점을 말하고 싶음",
    ],
  },
  {
    title: "이 연구에서 새롭게 제안하는 것",
    subtitle: "method / structure / evidence",
    bullets: [
      "dense O(N) action scoring 대신 orthogonal O(log N) routing 관점 제안",
      "learnable orthogonal projection layer 제안",
      "scaling / ablation / entropy / tracking / transplant 실험으로 다각도 검증",
      "speedup 숫자만이 아니라 내부 메커니즘도 같이 설명",
    ],
    notes: ["아이디어와 수식, 그리고 실험 해석이 같이 묶여 있다는 점이 강점"],
  },
  {
    title: "논문으로 쓰면 이런 구조",
    subtitle: "메시지도 비교적 깔끔하게 정리 가능",
    bullets: [
      "1. Introduction",
      "2. Related Work",
      "3. Method",
      "4. Why log2(N) / Why Orthogonality",
      "5. Experiments",
      "6. Discussion / Limitation",
      "7. Conclusion",
    ],
    notes: ["문제 제기, 이론적 동기, 방법, 실험, 해석으로 무리 없이 이어짐"],
  },
  {
    title: "마지막으로 하고 싶은 말",
    subtitle: "이건 단순한 head speedup 얘기만은 아님",
    bullets: [
      "action inference를 다른 방식으로 볼 수 있는가",
      "large action regime에서 routing 구조를 바꿀 가치가 있는가",
      "현재 결과들은 그 가능성을 꽤 강하게 보여주는 편",
    ],
    notes: [
      "GVLA의 포인트는 더 빠른 head 하나를 만드는 것만은 아님",
      "action inference를 dense comparison 말고 다른 방식으로 짤 수 있는지 보여보자는 데 있음",
    ],
  },
];

function addText(slide, text, opts) {
  slide.addText(text, Object.assign({
    fontFace: "Aptos",
    color: COLORS.ink,
    margin: 0,
  }, opts));
}

function addHeader(slide, title, subtitle, idx, total) {
  slide.addShape(ppt.ShapeType.rect, {
    x: 0, y: 0, w: 13.333, h: 0.7,
    fill: { color: COLORS.navy }, line: { color: COLORS.navy },
  });
  addText(slide, title, { x: 0.45, y: 0.16, w: 9.5, h: 0.28, fontSize: 26, bold: true, color: COLORS.white });
  addText(slide, subtitle, { x: 0.48, y: 0.78, w: 10.8, h: 0.28, fontSize: 11, color: COLORS.gray, italic: true });
  addText(slide, `${idx}/${total}`, { x: 12.2, y: 0.16, w: 0.7, h: 0.25, fontSize: 11, color: COLORS.white, align: "right" });
  slide.addShape(ppt.ShapeType.line, {
    x: 0.45, y: 1.13, w: 12.4, h: 0,
    line: { color: COLORS.line, pt: 1.2 },
  });
}

function addBullets(slide, items, box) {
  const runs = [];
  items.forEach((item) => {
    runs.push({
      text: item,
      options: { bullet: { indent: 14 }, hanging: 3, breakLine: true },
    });
  });
  slide.addText(runs, {
    x: box.x, y: box.y, w: box.w, h: box.h,
    fontFace: "Aptos",
    fontSize: box.fontSize || 18,
    color: COLORS.ink,
    paraSpaceAfterPt: 8,
    valign: "top",
    margin: 0.03,
  });
}

function addCode(slide, lines, box) {
  slide.addShape(ppt.ShapeType.roundRect, {
    x: box.x, y: box.y, w: box.w, h: box.h,
    rectRadius: 0.08,
    fill: { color: "F3F4F6" },
    line: { color: "E5E7EB", pt: 1 },
  });
  addText(slide, lines.join("\n"), {
    x: box.x + 0.18, y: box.y + 0.12, w: box.w - 0.3, h: box.h - 0.2,
    fontFace: "Courier New", fontSize: 17, color: COLORS.ink,
  });
}

function addTable(slide, table, box, small) {
  const rows = [table.headers].concat(table.rows);
  const data = rows.map((r, i) => r.map((c) => ({
    text: String(c),
    options: {
      bold: i === 0,
      color: i === 0 ? COLORS.white : COLORS.ink,
      fill: i === 0 ? COLORS.blue2 : (i % 2 ? "FFFFFF" : "F8FAFC"),
      align: "center",
      valign: "mid",
      margin: 0.03,
      fontSize: small ? 10 : 11,
      border: { pt: 0.5, color: COLORS.line },
    },
  })));
  slide.addTable(data, {
    x: box.x, y: box.y, w: box.w, h: box.h,
    fontFace: "Aptos",
    colW: table.headers.map(() => box.w / table.headers.length),
    rowH: small ? 0.3 : 0.34,
    border: { pt: 0.5, color: COLORS.line },
    fill: COLORS.white,
  });
}

function addNotes(slide, notes) {
  if (!notes || !notes.length) return;
  slide.addShape(ppt.ShapeType.roundRect, {
    x: 0.45, y: 6.6, w: 12.35, h: 0.55,
    rectRadius: 0.05,
    fill: { color: "EFF6FF" },
    line: { color: "BFDBFE", pt: 1 },
  });
  addText(slide, `멘트: ${notes.join(" / ")}`, {
    x: 0.62, y: 6.75, w: 12.0, h: 0.22, fontSize: 10.5, color: COLORS.blue, italic: true,
  });
}

async function buildPpt() {
  global.ppt = new PptxGenJS();
  ppt.layout = "LAYOUT_WIDE";
  ppt.author = "OpenAI Codex";
  ppt.company = "GVLA-Net";
  ppt.subject = "Professor proposal deck";
  ppt.title = "GVLA Professor Proposal";
  ppt.lang = "ko-KR";
  ppt.theme = {
    headFontFace: "Aptos",
    bodyFontFace: "Aptos",
    lang: "ko-KR",
  };

  slides.forEach((cfg, idx) => {
    const slide = ppt.addSlide();
    slide.background = { color: COLORS.white };

    if (cfg.kind === "title") {
      slide.addShape(ppt.ShapeType.rect, { x: 0, y: 0, w: 13.333, h: 7.5, fill: { color: COLORS.navy }, line: { color: COLORS.navy } });
      slide.addShape(ppt.ShapeType.rect, { x: 7.7, y: 0, w: 5.633, h: 7.5, fill: { color: COLORS.blue2, transparency: 10 }, line: { color: COLORS.blue2, transparency: 100 } });
      addText(slide, cfg.title, { x: 0.7, y: 1.0, w: 6.5, h: 0.7, fontSize: 28, bold: true, color: COLORS.white });
      addText(slide, cfg.subtitle, { x: 0.7, y: 1.8, w: 7.4, h: 0.5, fontSize: 22, color: "D1D5DB" });
      addText(slide, cfg.kicker, { x: 0.7, y: 2.7, w: 7.6, h: 0.8, fontSize: 24, bold: true, color: "BFDBFE" });
      addBullets(slide, cfg.bullets, { x: 0.88, y: 4.0, w: 6.7, h: 1.7, fontSize: 22 });
      addText(slide, "GVLA-Net Proposal Deck", { x: 0.72, y: 6.7, w: 4, h: 0.25, fontSize: 11, color: "CBD5E1" });
      return;
    }

    addHeader(slide, cfg.title, cfg.subtitle || "", idx + 1, slides.length);

    if (cfg.twoColLists) {
      addText(slide, cfg.twoColLists.leftTitle, { x: 0.6, y: 1.45, w: 2.5, h: 0.25, fontSize: 17, bold: true, color: COLORS.blue });
      addText(slide, cfg.twoColLists.rightTitle, { x: 6.8, y: 1.45, w: 2.5, h: 0.25, fontSize: 17, bold: true, color: COLORS.red });
      addBullets(slide, cfg.twoColLists.left, { x: 0.7, y: 1.8, w: 5.1, h: 3.6, fontSize: 18 });
      addBullets(slide, cfg.twoColLists.right, { x: 6.9, y: 1.8, w: 5.3, h: 3.6, fontSize: 18 });
    } else if (cfg.imagePair) {
      slide.addImage({ path: cfg.imagePair[0], x: 0.55, y: 1.45, w: 6.0, h: 2.5 });
      slide.addImage({ path: cfg.imagePair[1], x: 6.75, y: 1.45, w: 6.0, h: 2.5 });
      if (cfg.bullets) addBullets(slide, cfg.bullets, { x: 0.72, y: 4.25, w: 11.8, h: 1.9, fontSize: 17 });
    } else if (cfg.image && cfg.table) {
      slide.addImage({ path: cfg.image, x: 0.6, y: 1.45, w: 6.2, h: 3.55, sizing: { type: "contain", x: 0.6, y: 1.45, w: 6.2, h: 3.55 } });
      addTable(slide, cfg.table, { x: 7.0, y: 1.45, w: 5.7, h: 3.65 }, true);
    } else if (cfg.image) {
      slide.addImage({ path: cfg.image, x: 6.9, y: 1.5, w: 5.8, h: 3.9, sizing: { type: "contain", x: 6.9, y: 1.5, w: 5.8, h: 3.9 } });
      if (cfg.bullets) addBullets(slide, cfg.bullets, { x: 0.7, y: 1.55, w: 5.7, h: 3.8, fontSize: 17 });
      if (cfg.table) addTable(slide, cfg.table, { x: 0.75, y: 4.9, w: 12.0, h: 1.2 }, true);
    } else if (cfg.table && cfg.bullets) {
      addBullets(slide, cfg.bullets, { x: 0.72, y: 1.45, w: 12.0, h: 1.6, fontSize: 17 });
      addTable(slide, cfg.table, { x: 0.72, y: 3.0, w: 12.0, h: 2.9 }, true);
    } else if (cfg.table) {
      addTable(slide, cfg.table, { x: 0.72, y: 1.55, w: 12.0, h: 4.9 }, true);
    } else if (cfg.code && cfg.bullets) {
      addCode(slide, cfg.code, { x: 0.72, y: 1.55, w: 5.4, h: 2.1 });
      addBullets(slide, cfg.bullets, { x: 6.45, y: 1.52, w: 6.0, h: 3.3, fontSize: 17 });
    } else if (cfg.code) {
      addCode(slide, cfg.code, { x: 0.85, y: 1.7, w: 5.0, h: 2.2 });
      if (cfg.bullets) addBullets(slide, cfg.bullets, { x: 6.1, y: 1.7, w: 6.0, h: 3.0 });
    } else if (cfg.bullets) {
      addBullets(slide, cfg.bullets, { x: 0.72, y: 1.55, w: 12.0, h: 4.4, fontSize: 18 });
    }
    addNotes(slide, cfg.notes);
  });

  await ppt.writeFile({ fileName: PPTX_OUT });
}

function wrapText(text, maxChars) {
  const words = text.split(" ");
  const lines = [];
  let cur = "";
  words.forEach((w) => {
    if ((cur + " " + w).trim().length > maxChars) {
      if (cur) lines.push(cur.trim());
      cur = w;
    } else {
      cur += ` ${w}`;
    }
  });
  if (cur.trim()) lines.push(cur.trim());
  return lines;
}

async function buildPdf() {
  const pdf = await PDFDocument.create();
  pdf.registerFontkit(fontkit);
  const fontBytes = fs.readFileSync(FONT_PATH);
  const krFont = await pdf.embedFont(fontBytes, { subset: false });
  const mono = await pdf.embedFont(StandardFonts.Courier);

  for (let i = 0; i < slides.length; i++) {
    const cfg = slides[i];
    const page = pdf.addPage([960, 540]);
    const { width, height } = page.getSize();

    page.drawRectangle({ x: 0, y: height - 46, width, height: 46, color: rgb(15/255, 23/255, 42/255) });
    page.drawText(cfg.kind === "title" ? cfg.title : cfg.title, { x: 32, y: height - 31, size: cfg.kind === "title" ? 24 : 20, font: krFont, color: rgb(1,1,1) });
    if (cfg.kind !== "title") {
      page.drawText(cfg.subtitle || "", { x: 34, y: height - 60, size: 10.5, font: krFont, color: rgb(107/255,114/255,128/255) });
      page.drawText(`${i + 1}/${slides.length}`, { x: width - 52, y: height - 30, size: 10, font: krFont, color: rgb(1,1,1) });
    }

    if (cfg.kind === "title") {
      page.drawRectangle({ x: 0, y: 0, width, height, color: rgb(15/255,23/255,42/255) });
      page.drawRectangle({ x: 560, y: 0, width: 400, height, color: rgb(37/255,99/255,235/255), opacity: 0.92 });
      page.drawText(cfg.title, { x: 48, y: 390, size: 30, font: krFont, color: rgb(1,1,1) });
      page.drawText(cfg.subtitle, { x: 48, y: 352, size: 20, font: krFont, color: rgb(209/255,213/255,219/255) });
      page.drawText(cfg.kicker, { x: 48, y: 298, size: 22, font: krFont, color: rgb(191/255,219/255,254/255), maxWidth: 500 });
      let y = 215;
      cfg.bullets.forEach((b) => {
        page.drawText(`• ${b}`, { x: 56, y, size: 18, font: krFont, color: rgb(1,1,1) });
        y -= 30;
      });
      continue;
    }

    let cursorY = height - 95;
    const leftX = 36;

    if (cfg.twoColLists) {
      page.drawText(cfg.twoColLists.leftTitle, { x: leftX, y: cursorY, size: 16, font: krFont, color: rgb(29/255,78/255,216/255) });
      page.drawText(cfg.twoColLists.rightTitle, { x: 480, y: cursorY, size: 16, font: krFont, color: rgb(185/255,28/255,28/255) });
      let y1 = cursorY - 28;
      cfg.twoColLists.left.forEach((b) => { page.drawText(`• ${b}`, { x: leftX + 8, y: y1, size: 15, font: krFont, color: rgb(17/255,24/255,39/255), maxWidth: 360 }); y1 -= 26; });
      let y2 = cursorY - 28;
      cfg.twoColLists.right.forEach((b) => { page.drawText(`• ${b}`, { x: 488, y: y2, size: 15, font: krFont, color: rgb(17/255,24/255,39/255), maxWidth: 360 }); y2 -= 26; });
      cursorY = 120;
    } else if (cfg.imagePair) {
      const img1 = await pdf.embedPng(fs.readFileSync(cfg.imagePair[0]));
      const img2 = await pdf.embedPng(fs.readFileSync(cfg.imagePair[1]));
      page.drawImage(img1, { x: 36, y: 250, width: 400, height: 170 });
      page.drawImage(img2, { x: 500, y: 250, width: 400, height: 170 });
      let y = 210;
      (cfg.bullets || []).forEach((b) => {
        page.drawText(`• ${b}`, { x: 44, y, size: 14, font: krFont, color: rgb(17/255,24/255,39/255), maxWidth: 860 });
        y -= 24;
      });
      cursorY = 70;
    } else if (cfg.image && cfg.table) {
      const img = await pdf.embedPng(fs.readFileSync(cfg.image));
      page.drawImage(img, { x: 36, y: 195, width: 450, height: 250 });
      drawSimpleTable(page, krFont, 510, 205, 410, cfg.table, true);
      cursorY = 90;
    } else if (cfg.image) {
      const img = await pdf.embedPng(fs.readFileSync(cfg.image));
      page.drawImage(img, { x: 490, y: 180, width: 400, height: 260 });
      let y = cursorY - 12;
      (cfg.bullets || []).forEach((b) => {
        page.drawText(`• ${b}`, { x: leftX, y, size: 15, font: krFont, color: rgb(17/255,24/255,39/255), maxWidth: 400 });
        y -= 24;
      });
      if (cfg.table) drawSimpleTable(page, krFont, 36, 52, 860, cfg.table, true);
      cursorY = 90;
    } else if (cfg.table && cfg.bullets) {
      let y = cursorY - 12;
      cfg.bullets.forEach((b) => {
        page.drawText(`• ${b}`, { x: leftX, y, size: 15, font: krFont, color: rgb(17/255,24/255,39/255), maxWidth: 860 });
        y -= 24;
      });
      drawSimpleTable(page, krFont, 36, 86, 860, cfg.table, true);
      cursorY = 68;
    } else if (cfg.table) {
      drawSimpleTable(page, krFont, 36, 120, 860, cfg.table, true);
      cursorY = 96;
    } else if (cfg.code && cfg.bullets) {
      page.drawRectangle({ x: 36, y: 255, width: 360, height: 145, color: rgb(243/255,244/255,246/255), borderColor: rgb(229/255,231/255,235/255), borderWidth: 1 });
      page.drawText(cfg.code.join("\n"), { x: 52, y: 368, size: 13, font: mono, color: rgb(17/255,24/255,39/255), lineHeight: 18 });
      let y = 390;
      cfg.bullets.forEach((b) => {
        page.drawText(`• ${b}`, { x: 430, y, size: 15, font: krFont, color: rgb(17/255,24/255,39/255), maxWidth: 460 });
        y -= 24;
      });
      cursorY = 90;
    } else {
      let y = cursorY - 12;
      (cfg.bullets || []).forEach((b) => {
        page.drawText(`• ${b}`, { x: leftX, y, size: 16, font: krFont, color: rgb(17/255,24/255,39/255), maxWidth: 860 });
        y -= 28;
      });
      cursorY = y;
    }

    if (cfg.notes && cfg.notes.length) {
      page.drawRectangle({ x: 30, y: 22, width: 900, height: 40, color: rgb(239/255,246/255,255/255), borderColor: rgb(191/255,219/255,254/255), borderWidth: 1 });
      const noteText = `멘트: ${cfg.notes.join(" / ")}`;
      const noteLines = wrapText(noteText, 86).slice(0, 2);
      let ny = 45;
      noteLines.forEach((line) => {
        page.drawText(line, { x: 42, y: ny, size: 9.5, font: krFont, color: rgb(29/255,78/255,216/255) });
        ny -= 12;
      });
    }
  }

  fs.writeFileSync(PDF_OUT, await pdf.save());
}

function drawSimpleTable(page, font, x, yBottom, w, table, compact) {
  const headers = table.headers;
  const rows = table.rows;
  const rowH = compact ? 22 : 26;
  const totalRows = rows.length + 1;
  const yTop = yBottom + totalRows * rowH;
  const colW = w / headers.length;

  page.drawRectangle({ x, y: yTop - rowH, width: w, height: rowH, color: rgb(37/255,99/255,235/255) });
  for (let c = 0; c < headers.length; c++) {
    page.drawText(String(headers[c]), { x: x + c * colW + 4, y: yTop - rowH + 6, size: 9.5, font, color: rgb(1,1,1), maxWidth: colW - 8 });
  }
  rows.forEach((row, r) => {
    const y = yTop - rowH * (r + 2);
    page.drawRectangle({ x, y, width: w, height: rowH, color: rgb(r % 2 ? 1 : 248/255, r % 2 ? 1 : 250/255, r % 2 ? 1 : 252/255) });
    for (let c = 0; c < headers.length; c++) {
      page.drawText(String(row[c]), { x: x + c * colW + 4, y: y + 6, size: 9, font, color: rgb(17/255,24/255,39/255), maxWidth: colW - 8 });
    }
  });
  for (let c = 0; c <= headers.length; c++) {
    page.drawLine({ start: { x: x + c * colW, y: yBottom }, end: { x: x + c * colW, y: yTop }, thickness: 0.6, color: rgb(203/255,213/255,225/255) });
  }
  for (let r = 0; r <= totalRows; r++) {
    page.drawLine({ start: { x, y: yBottom + r * rowH }, end: { x: x + w, y: yBottom + r * rowH }, thickness: 0.6, color: rgb(203/255,213/255,225/255) });
  }
}

async function main() {
  if (!fs.existsSync(FONT_PATH)) {
    throw new Error(`Missing font: ${FONT_PATH}`);
  }
  await buildPpt();
  await buildPdf();
  console.log(`Wrote ${PPTX_OUT}`);
  console.log(`Wrote ${PDF_OUT}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
