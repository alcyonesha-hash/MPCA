/**
 * Survey Application
 *
 * Features:
 * - A/B randomization (which is shown as A vs B)
 * - Set order randomization
 * - Attention check insertion
 * - Google Sheets data submission
 */

class SurveyApp {
    constructor() {
        this.currentScreen = 'consent';
        this.currentSetIndex = 0;
        this.responses = [];
        this.participant = {};
        this.startTime = null;
        this.setStartTime = null;

        // Fixed A/B order for each set (no randomization)
        // true = swap A/B positions, false = keep original order
        this.abRandomization = SURVEY_SETS.map(() => false);

        // Fixed set order (no randomization)
        this.setOrder = this.generateSetOrder();

        // Track answers for current set
        this.currentAnswers = {};

        this.init();
    }

    init() {
        this.bindEvents();
        this.showScreen('consent');
    }

    generateSetOrder() {
        // Fixed order: sets 0-9 in sequence, attention check after set 5
        const order = [];
        for (let i = 0; i < SURVEY_SETS.length; i++) {
            order.push({ type: 'survey', index: i });
            if (i === 4) {
                order.push({ type: 'attention' });
            }
        }
        return order;
    }

    bindEvents() {
        // Consent
        const consentCheckbox = document.getElementById('consent-checkbox');
        const consentBtn = document.getElementById('consent-btn');

        consentCheckbox.addEventListener('change', (e) => {
            consentBtn.disabled = !e.target.checked;
        });

        consentBtn.addEventListener('click', () => {
            this.showScreen('info');
        });

        // Participant Info
        const infoBtn = document.getElementById('info-btn');
        const genderSelect = document.getElementById('participant-gender');
        const ageSelect = document.getElementById('participant-age');
        const techSelect = document.getElementById('participant-tech');
        const aiSelect = document.getElementById('participant-ai');

        const validateInfo = () => {
            const valid = genderSelect.value &&
                         ageSelect.value &&
                         techSelect.value &&
                         aiSelect.value;
            infoBtn.disabled = !valid;
        };

        [genderSelect, ageSelect, techSelect, aiSelect].forEach(el => {
            el.addEventListener('change', validateInfo);
        });

        infoBtn.addEventListener('click', () => {
            const phoneInput = document.getElementById('participant-phone');
            this.participant = {
                gender: genderSelect.value,
                ageGroup: ageSelect.value,
                techExperience: techSelect.value,
                aiExperience: aiSelect.value,
                phone: phoneInput ? phoneInput.value.trim() : '',
                startTime: new Date().toISOString()
            };
            this.startTime = Date.now();
            this.showScreen('description');
        });

        // Experiment Description
        document.getElementById('description-btn').addEventListener('click', () => {
            this.showScreen('instructions');
        });

        // Instructions
        document.getElementById('instructions-btn').addEventListener('click', () => {
            this.showScreen('survey');
            this.loadCurrentSet();
        });

        // Option buttons
        document.querySelectorAll('.option-btn[data-question]').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const question = e.target.dataset.question;
                const value = e.target.dataset.value;
                this.selectOption(question, value);
            });
        });

        // Attention check buttons
        document.querySelectorAll('.attention-option').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const value = e.target.dataset.value;
                this.handleAttentionCheck(value);
            });
        });

        // Next button
        document.getElementById('next-btn').addEventListener('click', () => {
            this.saveCurrentResponse();
            this.nextSet();
        });
    }

    showScreen(screenId) {
        document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
        document.getElementById(`${screenId}-screen`).classList.add('active');
        this.currentScreen = screenId;
    }

    loadCurrentSet() {
        const setInfo = this.setOrder[this.currentSetIndex];

        if (setInfo.type === 'attention') {
            this.showScreen('attention');
            return;
        }

        const setIndex = setInfo.index;
        const set = SURVEY_SETS[setIndex];
        const swapAB = this.abRandomization[setIndex];

        // Update progress
        const surveyProgress = this.setOrder.filter((s, i) => i <= this.currentSetIndex && s.type === 'survey').length;
        const totalSurveys = SURVEY_SETS.length;
        document.getElementById('progress-fill').style.width = `${(surveyProgress / totalSurveys) * 100}%`;
        document.getElementById('progress-text').textContent = `${surveyProgress} / ${totalSurveys}`;

        // Load chat content
        const fullChat = set.full;
        const comparisonChat = set.comparison;

        const chatA = swapAB ? comparisonChat : fullChat;
        const chatB = swapAB ? fullChat : comparisonChat;

        this.renderChat('chat-a', chatA, set.type);
        this.renderChat('chat-b', chatB, set.type);

        // Add single replay button for GIF sets
        if (set.type === 'gif') {
            this.addGifReplayButton(chatA.src, chatB.src);
        }

        // Reset answers
        this.currentAnswers = {};
        document.querySelectorAll('.option-btn[data-question]').forEach(btn => {
            btn.classList.remove('selected');
        });
        document.getElementById('next-btn').disabled = true;

        // Store metadata
        this.currentSetMeta = {
            setIndex: setIndex,
            setType: set.type,
            comparisonType: set.comparisonType,
            swapAB: swapAB,
            fullPosition: swapAB ? 'B' : 'A'
        };

        this.setStartTime = Date.now();
    }

    renderChat(containerId, messages, type) {
        const container = document.getElementById(containerId);

        if (type === 'gif') {
            // GIF display - show placeholder until Play button clicked
            // Use a canvas to show first frame (static) instead of GIF
            container.innerHTML = `
                <div class="gif-container">
                    <div class="gif-placeholder" id="${containerId}-placeholder">
                        <span class="play-icon">▶</span>
                        <span class="play-text">Click Play to start</span>
                    </div>
                    <img src="" alt="Chat animation" id="${containerId}-gif" class="chat-gif hidden" data-src="${messages.src}">
                </div>
            `;
            // Store the source for replay
            container.dataset.gifSrc = messages.src;
        } else {
            // Text chat display
            let html = '';
            messages.forEach(msg => {
                const isAgent = msg.role === 'agent';
                html += `
                    <div class="chat-message ${msg.role}">
                        <span class="message-sender">${msg.sender || (isAgent ? 'Agent' : 'User')}</span>
                        <div class="message-bubble">${msg.text}</div>
                        ${msg.time ? `<span class="message-time">${msg.time}</span>` : ''}
                    </div>
                `;
            });
            container.innerHTML = html;
        }
    }

    addGifReplayButton(srcA, srcB) {
        // Remove existing replay button if any
        const existingBtn = document.querySelector('.gif-replay-btn');
        if (existingBtn) existingBtn.remove();

        // Add single replay button between the panels
        const comparisonContainer = document.querySelector('.comparison-container');
        const replayBtn = document.createElement('button');
        replayBtn.className = 'btn primary gif-replay-btn';
        replayBtn.innerHTML = 'Play Both<span class="ko">재생</span>';
        replayBtn.onclick = () => {
            // Hide placeholders and show GIFs
            const placeholderA = document.getElementById('chat-a-placeholder');
            const placeholderB = document.getElementById('chat-b-placeholder');
            const imgA = document.getElementById('chat-a-gif');
            const imgB = document.getElementById('chat-b-gif');

            if (placeholderA) placeholderA.classList.add('hidden');
            if (placeholderB) placeholderB.classList.add('hidden');

            // Load and play both GIFs simultaneously with cache-busting
            const timestamp = Date.now();
            if (imgA) {
                imgA.classList.remove('hidden');
                imgA.src = srcA + '?t=' + timestamp;
            }
            if (imgB) {
                imgB.classList.remove('hidden');
                imgB.src = srcB + '?t=' + timestamp;
            }
        };
        comparisonContainer.parentNode.insertBefore(replayBtn, comparisonContainer.nextSibling);
    }

    selectOption(question, value) {
        // Update UI
        document.querySelectorAll(`.option-btn[data-question="${question}"]`).forEach(btn => {
            btn.classList.remove('selected');
        });
        document.querySelector(`.option-btn[data-question="${question}"][data-value="${value}"]`).classList.add('selected');

        // Store answer
        this.currentAnswers[question] = value;

        // Check if all questions answered
        const allAnswered = ['q1', 'q2', 'q3', 'q4'].every(q => this.currentAnswers[q]);
        document.getElementById('next-btn').disabled = !allAnswered;
    }

    handleAttentionCheck(value) {
        document.querySelectorAll('.attention-option').forEach(btn => {
            btn.classList.remove('selected');
        });
        document.querySelector(`.attention-option[data-value="${value}"]`).classList.add('selected');

        // Record attention check result
        this.participant.attentionCheck = value;
        this.participant.attentionCheckPassed = (value === 'A');

        // Move to next set after brief delay
        setTimeout(() => {
            this.currentSetIndex++;
            if (this.currentSetIndex < this.setOrder.length) {
                this.showScreen('survey');
                this.loadCurrentSet();
            } else {
                this.completesurvey();
            }
        }, 500);
    }

    saveCurrentResponse() {
        const meta = this.currentSetMeta;
        const timeSpent = Date.now() - this.setStartTime;

        // Convert answers to reference Full condition
        // If swapAB is true, Full is B, so we need to flip the answers
        const convertAnswer = (answer) => {
            if (meta.swapAB) {
                return answer === 'A' ? 'B' : 'A';
            }
            return answer;
        };

        const response = {
            setIndex: meta.setIndex,
            setType: meta.setType,
            comparisonType: meta.comparisonType,
            fullPosition: meta.fullPosition,

            // Raw answers (as shown to user)
            rawQ1: this.currentAnswers.q1,
            rawQ2: this.currentAnswers.q2,
            rawQ3: this.currentAnswers.q3,
            rawQ4: this.currentAnswers.q4,

            // Normalized answers (relative to Full condition)
            // "A" means user chose Full, "B" means user chose comparison
            q1_humanLike: convertAnswer(this.currentAnswers.q1) === 'A' ? 'Full' : meta.comparisonType,
            q2_lessAnnoying: convertAnswer(this.currentAnswers.q2) === 'A' ? 'Full' : meta.comparisonType,
            q3_naturalFlow: convertAnswer(this.currentAnswers.q3) === 'A' ? 'Full' : meta.comparisonType,
            q4_keepInChat: convertAnswer(this.currentAnswers.q4) === 'A' ? 'Full' : meta.comparisonType,

            // Did user prefer Full?
            prefersFullQ1: convertAnswer(this.currentAnswers.q1) === 'A',
            prefersFullQ2: convertAnswer(this.currentAnswers.q2) === 'A',
            prefersFullQ3: convertAnswer(this.currentAnswers.q3) === 'A',
            prefersFullQ4: convertAnswer(this.currentAnswers.q4) === 'A',

            timeSpentMs: timeSpent
        };

        this.responses.push(response);
    }

    nextSet() {
        this.currentSetIndex++;

        if (this.currentSetIndex < this.setOrder.length) {
            const setInfo = this.setOrder[this.currentSetIndex];
            if (setInfo.type === 'attention') {
                this.showScreen('attention');
            } else {
                this.loadCurrentSet();
            }
        } else {
            this.completeSurvey();
        }
    }

    completeSurvey() {
        const totalTime = Date.now() - this.startTime;

        const surveyData = {
            participant: this.participant,
            responses: this.responses,
            metadata: {
                totalTimeMs: totalTime,
                completedAt: new Date().toISOString(),
                setOrder: this.setOrder.map(s => s.type === 'attention' ? 'attention' : s.index),
                abRandomization: this.abRandomization
            }
        };

        // Send to Google Sheets
        this.submitToGoogleSheets(surveyData);

        // Show completion screen
        this.showScreen('complete');
    }

    async submitToGoogleSheets(data) {
        // Google Sheets Web App URL (to be configured)
        const GOOGLE_SHEETS_URL = window.GOOGLE_SHEETS_WEBHOOK_URL || '';

        if (!GOOGLE_SHEETS_URL) {
            console.log('Google Sheets URL not configured. Data:', data);
            // Save locally as backup
            this.downloadAsJSON(data);
            return;
        }

        try {
            const response = await fetch(GOOGLE_SHEETS_URL, {
                method: 'POST',
                mode: 'no-cors',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data)
            });
            console.log('Data submitted to Google Sheets');
        } catch (error) {
            console.error('Failed to submit to Google Sheets:', error);
            // Save locally as backup
            this.downloadAsJSON(data);
        }
    }

    downloadAsJSON(data) {
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `survey_response_${data.participant.email}_${Date.now()}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.surveyApp = new SurveyApp();
});
