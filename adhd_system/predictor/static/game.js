let currentColor;
let gameRunning = false;
let clickable = false;
let totalGreen = 0;
let missedGreen = 0;
let totalRed = 0;
let clickedRed = 0;
let stimulusStartTime;
let reactionTimes = [];
let intervals = [];

const box = document.getElementById("box");
const statusText = document.getElementById("status");
const startBtn = document.getElementById("startBtn");
const instructionBox = document.getElementById("instructions");
const flashOverlay = document.getElementById("flash-overlay");

function startGame() {
    startBtn.style.display = "none";
    instructionBox.style.display = "none";

    totalGreen = 0;
    missedGreen = 0;
    totalRed = 0;
    clickedRed = 0;
    reactionTimes = [];
    intervals = [];

    gameRunning = true;
    statusText.innerText = "Sustained Attention Task In Progress...";
    statusText.style.color = "var(--text-main)";

    startDistractors();
    nextStimulus();

    // Test duration: 45 seconds for clinical validity
    setTimeout(endGame, 45000);
}

function startDistractors() {
    // Drifting shapes every 2 seconds
    const distractorTimer = setInterval(() => {
        if (!gameRunning) {
            clearInterval(distractorTimer);
            return;
        }
        createDriftingShape();
    }, 2000);

    // Subtle edge flashes
    const flashTimer = setInterval(() => {
        if (!gameRunning) {
            clearInterval(flashTimer);
            return;
        }
        if (Math.random() < 0.3) {
            triggerEdgeFlash();
        }
    }, 5000);
}

function createDriftingShape() {
    const shape = document.createElement("div");
    const isTriangle = Math.random() > 0.5;
    shape.className = `distractor-shape ${isTriangle ? 'triangle' : 'circle'}`;

    const size = Math.random() * 40 + 20;
    if (!isTriangle) {
        shape.style.width = size + "px";
        shape.style.height = size + "px";
    }

    // Spawn at edges
    const side = Math.floor(Math.random() * 4);
    let x, y;
    if (side === 0) { x = -50; y = Math.random() * window.innerHeight; }
    else if (side === 1) { x = window.innerWidth + 50; y = Math.random() * window.innerHeight; }
    else if (side === 2) { y = -50; x = Math.random() * window.innerWidth; }
    else { y = window.innerHeight + 50; x = Math.random() * window.innerWidth; }

    shape.style.left = x + "px";
    shape.style.top = y + "px";
    document.body.appendChild(shape);

    // Target a point on the opposite side, avoiding the center (600x400 area)
    const targetX = Math.random() * window.innerWidth;
    const targetY = Math.random() * window.innerHeight;

    // Slow drift animation
    const duration = Math.random() * 10000 + 10000;
    shape.animate([
        { transform: `translate(0, 0)` },
        { transform: `translate(${targetX - x}px, ${targetY - y}px)` }
    ], {
        duration: duration,
        easing: 'linear'
    });

    setTimeout(() => shape.remove(), duration);
}

function triggerEdgeFlash() {
    flashOverlay.style.opacity = "0.05";
    setTimeout(() => {
        flashOverlay.style.opacity = "0";
    }, 150);
}

function nextStimulus() {
    if (!gameRunning) return;

    box.style.display = "none";
    clickable = false;

    // Random ISI: 0.8s to 1.5s
    const delay = Math.random() * 700 + 800;
    setTimeout(() => {
        if (!gameRunning) return;

        // 75/25 distribution
        currentColor = Math.random() < 0.75 ? "green" : "red";
        box.style.backgroundColor = currentColor === "green" ? "#10b981" : "#ef4444";
        box.style.display = "block";

        stimulusStartTime = Date.now();
        clickable = true;

        if (currentColor === "green") totalGreen++;
        else totalRed++;

        // Random stimulus duration: 600ms to 900ms
        const visibilityDuration = Math.random() * 300 + 600;
        setTimeout(() => {
            if (clickable && gameRunning) {
                if (currentColor === "green") {
                    missedGreen++;
                }
                nextStimulus();
            }
        }, visibilityDuration);

    }, delay);
}

box.onclick = () => {
    if (!clickable || !gameRunning) return;

    const reactionTime = Date.now() - stimulusStartTime;
    clickable = false;
    box.style.display = "none";

    if (currentColor === "green") {
        reactionTimes.push(reactionTime);
    } else if (currentColor === "red") {
        clickedRed++;
    }

    nextStimulus();
};

function calculateStdDev(array) {
    if (array.length === 0) return 0;
    const n = array.length;
    const mean = array.reduce((a, b) => a + b) / n;
    return Math.sqrt(array.map(x => Math.pow(x - mean, 2)).reduce((a, b) => a + b) / n).toFixed(2);
}

const getScore = (rate) => {
    if (rate <= 0.05) return 0;
    if (rate <= 0.15) return 1;
    if (rate <= 0.30) return 2;
    if (rate <= 0.50) return 3;
    if (rate <= 0.70) return 4;
    return 5;
};

function endGame() {
    gameRunning = false;
    box.style.display = "none";

    const inattentionRate = totalGreen > 0 ? missedGreen / totalGreen : 0;
    const impulsivityRate = totalRed > 0 ? clickedRed / totalRed : 0;

    const inScore = getScore(inattentionRate);
    const imScore = getScore(impulsivityRate);

    // RT Variability is the SD of reaction times
    const rtVariability = calculateStdDev(reactionTimes);

    if (window.opener && !window.opener.closed) {
        try {
            window.opener.document.getElementById("InattentionScore").value = inScore;
            window.opener.document.getElementById("ImpulsivityScore").value = imScore;

            window.opener.document.getElementById("total_trials").value = totalGreen + totalRed;
            window.opener.document.getElementById("correct_go").value = totalGreen - missedGreen;
            window.opener.document.getElementById("missed_go").value = missedGreen;
            window.opener.document.getElementById("correct_inhibit").value = totalRed - clickedRed;
            window.opener.document.getElementById("commission_errors").value = clickedRed;
            window.opener.document.getElementById("reaction_times").value = JSON.stringify(reactionTimes);

            // New variability metric
            if (window.opener.document.getElementById("rt_variability")) {
                window.opener.document.getElementById("rt_variability").value = rtVariability;
            }

            const parentBtn = window.opener.document.querySelector("button[onclick='openGame()']");
            if (parentBtn) {
                parentBtn.innerText = "Assessment Complete ✓";
                parentBtn.style.color = "#10b981";
                parentBtn.style.borderColor = "#10b981";
            }
        } catch (e) { console.error(e); }
    }

    statusText.innerText = "Assessment Finished";
    statusText.style.color = "#10b981";
    setTimeout(() => window.close(), 3000);
}
