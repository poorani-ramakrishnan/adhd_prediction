const synth = window.speechSynthesis;
let currentQuestionIndex = 0;
let voiceEnabled = false;
let questions = [];
let currentUtterance = null; // Prevent GC

document.addEventListener('DOMContentLoaded', () => {
    // Collect all question groups
    questions = Array.from(document.querySelectorAll('.question-group'));

    // Add event listeners to all inputs/selects to advance voice
    questions.forEach((group, index) => {
        const inputs = group.querySelectorAll('input, select');
        inputs.forEach(input => {
            // 'change' is good for selects and radios
            input.addEventListener('change', () => {
                if (voiceEnabled && index === currentQuestionIndex) {
                    handleInputChange(input, index);
                }
            });

            // For text/number inputs, handle enter key or blur
            if (input.type === 'number' || input.type === 'text') {
                input.addEventListener('keypress', (e) => {
                    if (e.key === 'Enter') {
                        e.preventDefault();
                        input.blur();
                        if (voiceEnabled && index === currentQuestionIndex) {
                            handleInputChange(input, index);
                        }
                    }
                });

                input.addEventListener('blur', () => {
                    if (voiceEnabled && index === currentQuestionIndex) {
                        handleInputChange(input, index);
                    }
                });
            }
        });
    });
});

function handleInputChange(input, index) {
    // Automation: Age to Education Stage
    if (input.id === 'age') {
        const age = parseInt(input.value);
        const eduSelect = document.getElementById('EducationStage');
        if (age && eduSelect) {
            if (age < 13) eduSelect.value = 'Child';
            else if (age < 19) eduSelect.value = 'Teen';
            else if (age < 23) eduSelect.value = 'Undergrad';
            else eduSelect.value = 'Adult';

            // Add visual feedback for auto-detection
            eduSelect.style.borderColor = 'var(--primary)';
            setTimeout(() => eduSelect.style.borderColor = '', 1000);
        }
    }

    // User requested: "after each user input move to the next question"
    // For Age (index 0), skip Education (index 1) and go to Gender (index 2)
    if (index === 0) {
        setTimeout(() => {
            readQuestion(2); // Jump directly to Gender after 1s gap
        }, 1000);
    } else {
        readNextQuestion();
    }
}

function speak(text, callback) {
    if (!voiceEnabled) return;

    // Cancel any ongoing speech
    synth.cancel();

    currentUtterance = new SpeechSynthesisUtterance(text);
    currentUtterance.rate = 1.0;
    currentUtterance.pitch = 1.0;

    const voices = synth.getVoices();
    const preferredVoice = voices.find(v => v.lang.includes('en-GB') || v.lang.includes('en-US')) || voices[0];
    if (preferredVoice) currentUtterance.voice = preferredVoice;

    if (callback) {
        currentUtterance.onend = callback;
    }

    synth.speak(currentUtterance);
}

function startVoice() {
    voiceEnabled = true;
    const btn = document.getElementById('voiceToggle');
    if (btn) btn.style.display = 'none';

    // Speak welcome and then start questions after 1s gap
    speak("Welcome to ADHD Predictor", () => {
        setTimeout(() => {
            readQuestion(0); // Start with Age
        }, 1000);
    });
}

function readQuestion(index) {
    if (index < questions.length) {
        currentQuestionIndex = index;

        const currentGroup = questions[index];
        const questionTextElement = currentGroup.querySelector('.question-text');

        if (questionTextElement) {
            const text = questionTextElement.innerText;

            // Visual focus
            questions.forEach(q => {
                q.style.border = 'none';
                q.style.background = 'transparent';
                q.style.boxShadow = 'none';
            });
            currentGroup.style.border = '2px solid var(--primary)';
            currentGroup.style.borderRadius = '16px';
            currentGroup.style.background = 'rgba(79, 70, 229, 0.05)';
            currentGroup.style.boxShadow = '0 0 20px rgba(79, 70, 229, 0.1)';
            currentGroup.scrollIntoView({ behavior: 'smooth', block: 'center' });

            speak(text);
        }
    }
}

function readNextQuestion() {
    // Check if current question is the last one
    if (currentQuestionIndex < questions.length - 1) {
        const nextIndex = currentQuestionIndex + 1;

        // Safety check: if the next question is EducationStage, and it's already filled, move to the one after
        const nextGroup = questions[nextIndex];
        const eduSelect = nextGroup.querySelector('#EducationStage');
        if (eduSelect && eduSelect.value !== "") {
            setTimeout(() => {
                readQuestion(nextIndex + 1); // Jump past it if somehow reached via readNext
            }, 1000);
            return;
        }

        setTimeout(() => {
            readQuestion(nextIndex);
        }, 1000); // Standard 1s gap after input
    }
}
