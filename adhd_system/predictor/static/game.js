let currentColor;
let gameRunning = false;
let clickable = false;
let totalGreen = 0;
let missedGreen = 0;
let totalRed = 0;
let clickedRed = 0;
let stimulusStartTime;
let reactionTimes = [];

const box = document.getElementById("box");
const statusText = document.getElementById("status");
const startBtn = document.getElementById("startBtn");
const instructionBox = document.getElementById("instructions");

function startGame() {
    startBtn.style.display = "none";
    instructionBox.style.display = "none";

    totalGreen = 0;
    missedGreen = 0;
    totalRed = 0;
    clickedRed = 0;
    reactionTimes = [];

    gameRunning = true;
    statusText.innerText = "Assessing... Stay Focused";
    statusText.style.color = "var(--text-main)";

    nextStimulus();
    // Test duration: 30 seconds
    setTimeout(endGame, 30000);
}

function nextStimulus() {
    if (!gameRunning) return;

    box.style.display = "none";
    clickable = false;

    // Random delay between stimuli
    setTimeout(() => {
        if (!gameRunning) return;

        // 70% chance of Green (Inattention test), 30% chance of Red (Impulsivity test)
        currentColor = Math.random() < 0.7 ? "green" : "red";

        // Use professional colors from CSS system or clean hex values
        box.style.backgroundColor = currentColor === "green" ? "#10b981" : "#ef4444";
        box.style.boxShadow = `0 0 40px ${currentColor === "green" ? "rgba(16, 185, 129, 0.4)" : "rgba(239, 68, 68, 0.4)"}`;

        box.style.display = "block";
        stimulusStartTime = Date.now();
        clickable = true;

        if (currentColor === "green") totalGreen++;
        else totalRed++;

        // Stimulus duration: 1 second
        setTimeout(() => {
            if (clickable && gameRunning) {
                if (currentColor === "green") {
                    missedGreen++;
                }
                nextStimulus();
            }
        }, 1000);

    }, Math.random() * 1000 + 800);
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

/**
 * Maps error rates to a 0-5 score for the prediction model
 */
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

    // Update parent window if it exists
    if (window.opener && !window.opener.closed) {
        try {
            window.opener.document.getElementById("InattentionScore").value = inScore;
            window.opener.document.getElementById("ImpulsivityScore").value = imScore;

            // Detailed stats for report
            window.opener.document.getElementById("total_trials").value = totalGreen + totalRed;
            window.opener.document.getElementById("correct_go").value = totalGreen - missedGreen;
            window.opener.document.getElementById("missed_go").value = missedGreen;
            window.opener.document.getElementById("correct_inhibit").value = totalRed - clickedRed;
            window.opener.document.getElementById("commission_errors").value = clickedRed;
            window.opener.document.getElementById("reaction_times").value = JSON.stringify(reactionTimes);

            // Visual feedback on parent window button
            const parentBtn = window.opener.document.querySelector("button[onclick='openGame()']");
            if (parentBtn) {
                parentBtn.innerText = "Assessment Complete ✓";
                parentBtn.style.color = "#10b981";
                parentBtn.style.borderColor = "#10b981";
            }
        } catch (e) {
            console.error("Could not update parent window:", e);
        }
    }

    statusText.innerText = "Assessment Complete";
    statusText.style.color = "#10b981";

    const summary = document.createElement("div");
    summary.style.marginTop = "20px";
    summary.style.color = "var(--text-muted)";
    summary.innerHTML = `<p>Test finished successfully.</p><p style="margin-top:10px; font-size:0.8rem;">Data captured. Closing in 3 seconds...</p>`;
    box.parentNode.appendChild(summary);

    setTimeout(() => window.close(), 3000);
}
