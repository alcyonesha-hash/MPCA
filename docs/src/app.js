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

        // Randomize A/B order for each set
        // true = swap A/B positions, false = keep original order
        this.abRandomization = SURVEY_SETS.map(() => Math.random() < 0.5);

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
        // Fixed order: sets 0-7 in sequence, attention check after set 4
        const order = [];
        for (let i = 0; i < SURVEY_SETS.length; i++) {
            order.push({ type: 'survey', index: i });
            if (i === 3) {
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

        // Reset scroll position to top
        document.getElementById('chat-a').scrollTop = 0;
        document.getElementById('chat-b').scrollTop = 0;

        // Update progress
        const surveyProgress = this.setOrder.filter((s, i) => i <= this.currentSetIndex && s.type === 'survey').length;
        const totalSurveys = SURVEY_SETS.length;
        document.getElementById('progress-fill').style.width = `${(surveyProgress / totalSurveys) * 100}%`;
        document.getElementById('progress-text').textContent = `${surveyProgress} / ${totalSurveys}`;

        // Update topic label
        const topicText = document.getElementById('topic-text');
        const topicKo = document.getElementById('topic-ko');
        if (topicText && set.topic) {
            topicText.textContent = set.topic;
        }
        if (topicKo && set.topicKo) {
            topicKo.textContent = set.topicKo;
        }

        // Load chat content
        const fullChat = set.full;
        const comparisonChat = set.comparison;

        const chatA = swapAB ? comparisonChat : fullChat;
        const chatB = swapAB ? fullChat : comparisonChat;

        this.renderChat('chat-a', chatA, set.type);
        this.renderChat('chat-b', chatB, set.type);

        // Add single replay button for video sets
        if (set.type === 'video') {
            this.addVideoReplayButton(chatA.src, chatB.src);
            this.addVideoNotice();
        } else {
            this.removeVideoNotice();
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

    // Avatar colors for consistent user coloring
    static AVATAR_COLORS = [
        '#4CAF50', '#2196F3', '#FF9800', '#9C27B0', '#00BCD4',
        '#FF5722', '#3F51B5', '#E91E63', '#8BC34A', '#795548'
    ];

    getAvatarColor(username, isAgent = false) {
        if (isAgent) return '#667eea';
        // Hash username to get consistent color
        let hash = 0;
        for (let i = 0; i < username.length; i++) {
            hash = username.charCodeAt(i) + ((hash << 5) - hash);
        }
        return SurveyApp.AVATAR_COLORS[Math.abs(hash) % SurveyApp.AVATAR_COLORS.length];
    }

    getInitials(username) {
        if (username.length <= 2) return username.toUpperCase();
        return username.substring(0, 2).toUpperCase();
    }

    renderChat(containerId, messages, type) {
        const container = document.getElementById(containerId);

        if (type === 'video') {
            // Video display - show ready message until Play button clicked
            container.innerHTML = `
                <div class="gif-container">
                    <div class="gif-ready-text" id="${containerId}-ready">
                        Click "Play Both" below<span class="ko">아래 "재생" 버튼을 누르세요</span>
                    </div>
                    <video id="${containerId}-video" class="chat-video hidden" data-src="${messages.src}" playsinline></video>
                </div>
            `;
            container.dataset.videoSrc = messages.src;
        } else {
            // Text chat display with avatars
            let html = '';
            messages.forEach(msg => {
                const isAgent = msg.role === 'agent';
                const sender = msg.sender || (isAgent ? 'Agent' : 'User');
                const avatarColor = this.getAvatarColor(sender, isAgent);
                const initials = this.getInitials(sender);

                // Split text into English and Korean parts (separated by \n)
                const parts = msg.text.split('\n');
                let textHtml = parts[0]; // English part
                if (parts.length > 1) {
                    // Add Korean part with special styling
                    textHtml += `<span class="message-ko">${parts[1]}</span>`;
                }
                html += `
                    <div class="chat-message ${msg.role}">
                        <div class="message-avatar" style="background-color: ${avatarColor}">${initials}</div>
                        <div class="message-content">
                            <span class="message-sender">${sender}</span>
                            <div class="message-bubble">${textHtml}</div>
                            ${msg.time ? `<span class="message-time">${msg.time}</span>` : ''}
                        </div>
                    </div>
                `;
            });
            container.innerHTML = html;
        }
    }

    addVideoReplayButton(srcA, srcB) {
        // Remove existing replay button if any
        const existingBtn = document.querySelector('.gif-replay-btn');
        if (existingBtn) existingBtn.remove();

        // Add single replay button between the panels
        const comparisonContainer = document.querySelector('.comparison-container');
        const replayBtn = document.createElement('button');
        replayBtn.className = 'btn primary gif-replay-btn';
        replayBtn.innerHTML = 'Play Both<span class="ko">재생</span>';
        replayBtn.onclick = () => {
            // Hide ready text and show videos
            const readyA = document.getElementById('chat-a-ready');
            const readyB = document.getElementById('chat-b-ready');
            const videoA = document.getElementById('chat-a-video');
            const videoB = document.getElementById('chat-b-video');

            if (readyA) readyA.classList.add('hidden');
            if (readyB) readyB.classList.add('hidden');

            // Load and play both videos simultaneously
            if (videoA) {
                videoA.classList.remove('hidden');
                videoA.src = srcA;
                videoA.currentTime = 0;
                videoA.play();
            }
            if (videoB) {
                videoB.classList.remove('hidden');
                videoB.src = srcB;
                videoB.currentTime = 0;
                videoB.play();
            }
        };
        comparisonContainer.parentNode.insertBefore(replayBtn, comparisonContainer.nextSibling);
    }

    addVideoNotice() {
        // Remove existing notice if any
        this.removeVideoNotice();

        // Add notice above the comparison container
        const comparisonContainer = document.querySelector('.comparison-container');
        const notice = document.createElement('div');
        notice.className = 'video-notice';
        notice.id = 'video-notice';
        notice.innerHTML = `
            ⏳ Watch the video until the end to see all agent responses
            <span class="ko">⏳ 영상을 끝까지 시청하여 모든 에이전트 응답을 확인하세요</span>
        `;
        comparisonContainer.parentNode.insertBefore(notice, comparisonContainer);
    }

    removeVideoNotice() {
        const existingNotice = document.getElementById('video-notice');
        if (existingNotice) existingNotice.remove();
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
                this.completeSurvey();
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
