let currentColor;
let gameRunning = false;
let clickable = false;
let totalGreen = 0;
let missedGreen = 0; 
let totalRed = 0;
let clickedRed = 0; 
const box = document.getElementById("box");
const statusText = document.getElementById("status");

function startGame() {
  startBtn.style.display = "none";
    totalGreen = 0; missedGreen = 0;
    totalRed = 0; clickedRed = 0; 
    gameRunning = true;
    statusText.innerText = "Game running... Focus!";
    nextStimulus();
    setTimeout(endGame, 30000); 
}

function nextStimulus() {
    if (!gameRunning) return;

    box.style.display = "none";
    clickable = false;
    setTimeout(() => {
        if (!gameRunning) return;

        currentColor = Math.random() < 0.7 ? "green" : "red";
        box.style.backgroundColor = currentColor;
        box.style.display = "block";
        clickable = true;

        if (currentColor === "green") totalGreen++;
        else totalRed++;

        setTimeout(() => {
            if (clickable) {
                if (currentColor === "green") missedGreen++; 
                nextStimulus();
            }
        }, 1000);

    }, Math.random() * 1000 + 500);
}

box.onclick = () => {
    if (!clickable) return;
    clickable = false;
    box.style.display = "none";

    if (currentColor === "red") {
        clickedRed++; 
    }
    
    nextStimulus(); 
};

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

    if (window.opener) {
        window.opener.document.getElementById("InattentionScore").value = inScore;
        window.opener.document.getElementById("ImpulsivityScore").value = imScore;
    }

    statusText.innerText = `Test complete. Inattention: ${inScore}, Impulsivity: ${imScore}`;
    setTimeout(() => window.close(), 2000);
}