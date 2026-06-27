document.addEventListener('DOMContentLoaded', () => {
    // Tab switching
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabPanes = document.querySelectorAll('.tab-pane');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            tabPanes.forEach(p => p.classList.remove('active'));
            
            btn.classList.add('active');
            document.getElementById(`tab-${btn.dataset.tab}`).classList.add('active');
        });
    });

    // Custom Toast Notification
    function showToast(message) {
        const container = document.getElementById('toast-container');
        const toast = document.createElement('div');
        toast.className = 'toast';
        
        const icon = document.createElement('div');
        icon.className = 'toast-icon';
        icon.textContent = '!';
        
        const textNode = document.createElement('span');
        textNode.textContent = message;
        
        toast.appendChild(icon);
        toast.appendChild(textNode);
        
        container.appendChild(toast);
        
        // Remove from DOM after animations finish (4s delay + 0.4s slide up)
        setTimeout(() => {
            toast.remove();
        }, 4400); 
    }

    // File Upload handling
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const changeImageBtn = document.getElementById('change-image-btn');
    const analyzeImageBtn = document.getElementById('analyze-image-btn');
    let selectedFile = null;
    let currentFileInput = fileInput; // always points to the active file input

    // Change Image button — stopPropagation prevents event bubbling to dropZone
    changeImageBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        currentFileInput.click();
    });

    dropZone.addEventListener('click', () => currentFileInput.click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFile(e.target.files[0]);
        }
    });

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            showToast('Please select an image file.');
            return;
        }
        selectedFile = file;
        changeImageBtn.classList.remove('hidden');
        analyzeImageBtn.classList.remove('hidden');

        // Show image preview inside the drop zone
        const reader = new FileReader();
        reader.onload = (e) => {
            dropZone.innerHTML = `
                <img src="${e.target.result}" alt="Selected image" id="image-preview"
                     style="max-width:100%; max-height:220px; border-radius:10px; object-fit:contain; display:block; margin:0 auto;">
                <input type="file" id="file-input" accept="image/*" hidden>
            `;
            // Update the shared reference to the new file input
            const newInput = dropZone.querySelector('#file-input');
            currentFileInput = newInput;
            newInput.addEventListener('change', (e) => {
                if (e.target.files.length) handleFile(e.target.files[0]);
            });
        };
        reader.readAsDataURL(file);
    }

    // ── Progress Bar ──
    const progressSection = document.getElementById('progress-section');
    const progressFill = document.getElementById('progress-fill');
    const progressTitle = document.getElementById('progress-title');
    const progressSteps = document.getElementById('progress-steps');
    let progressTimer = null;
    let progressStep = 0;

    function showProgress(title, steps) {
        progressStep = 0;
        progressTitle.textContent = title;
        progressSteps.innerHTML = steps.map((s, i) =>
            `<span class="step" data-step="${i}">${s}</span>`
        ).join('');
        progressSection.classList.remove('hidden');
        progressFill.style.width = '0%';
        // Mark first step active
        const first = progressSteps.querySelector('[data-step="0"]');
        if (first) first.classList.add('active');
    }

    function advanceProgress() {
        const steps = progressSteps.querySelectorAll('.step');
        if (progressStep < steps.length) {
            steps[progressStep]?.classList.remove('active');
            steps[progressStep]?.classList.add('done');
        }
        progressStep++;
        if (progressStep < steps.length) {
            steps[progressStep]?.classList.add('active');
        }
        const pct = Math.min(95, (progressStep / steps.length) * 100);
        progressFill.style.width = `${pct}%`;
    }

    function completeProgress() {
        if (progressTimer) { clearInterval(progressTimer); progressTimer = null; }
        const steps = progressSteps.querySelectorAll('.step');
        steps.forEach(s => { s.classList.remove('active'); s.classList.add('done'); });
        progressFill.style.width = '100%';
        progressTitle.textContent = 'Complete';
        setTimeout(() => {
            progressSection.classList.add('hidden');
        }, 800);
    }

    function hideProgress() {
        if (progressTimer) { clearInterval(progressTimer); progressTimer = null; }
        progressSection.classList.add('hidden');
    }

    function simulateProgress(title, steps, intervalMs = 3000) {
        showProgress(title, steps);
        if (progressTimer) clearInterval(progressTimer);
        progressTimer = setInterval(() => {
            if (progressStep < steps.length - 1) {
                advanceProgress();
            } else {
                clearInterval(progressTimer);
                progressTimer = null;
            }
        }, intervalMs);
    }

    // API Calls
    const analyzeTextBtn = document.getElementById('analyze-text-btn');
    const analyzeUrlBtn = document.getElementById('analyze-url-btn');

    analyzeTextBtn.addEventListener('click', async () => {
        const text = document.getElementById('text-input').value;
        if (!text.trim()) return showToast('Please enter some text.');
        
        // Check if the user pasted a single URL instead of text
        const urlPattern = /^(https?:\/\/[^\s]+)$/;
        if (urlPattern.test(text.trim())) {
            return showToast('It looks like you pasted a link! Please use the "URL" tab instead.');
        }

        simulateProgress('Fact-Checking...', ['Searching Web', 'Analyzing Content', 'Final Review'], 2500);
        await performAnalysis('/api/analyze/text', { text }, analyzeTextBtn);
    });

    analyzeUrlBtn.addEventListener('click', async () => {
        const url = document.getElementById('url-input').value;
        if (!url.trim() || !url.startsWith('http')) return showToast('Please enter a valid URL.');
        simulateProgress('Analyzing URL...', ['Fetching Page', 'Extracting Text', 'Analyzing Content', 'Final Review'], 2500);
        await performAnalysis('/api/analyze/url', { url }, analyzeUrlBtn);
    });

    analyzeImageBtn.addEventListener('click', async () => {
        if (!selectedFile) return showToast('No image selected.');
        
        const formData = new FormData();
        formData.append('image', selectedFile);

        // Send the optional user context (e.g. "this is an AI generated photo")
        const userContext = document.getElementById('image-context')?.value?.trim() || '';
        if (userContext) formData.append('context', userContext);

        setLoading(analyzeImageBtn, true);
        simulateProgress('Analyzing Image...', ['Face ID', 'AI Detection', 'Describing Image', 'Final Analysis'], 4000);
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 120000); // 2 min timeout
        try {
            const res = await fetch('/api/analyze/image', {
                method: 'POST',
                body: formData,
                signal: controller.signal
            });
            clearTimeout(timeoutId);
            const data = await res.json();
            if (data.error) throw new Error(data.error);
            completeProgress();
            renderResults(data);
        } catch (error) {
            clearTimeout(timeoutId);
            hideProgress();
            if (error.name === 'AbortError') {
                showToast('Analysis timed out. Please try again.');
            } else {
                showToast(error.message);
            }
        } finally {
            setLoading(analyzeImageBtn, false);
        }
    });

    async function performAnalysis(endpoint, payload, btn) {
        setLoading(btn, true);
        try {
            const res = await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            if (data.error) throw new Error(data.error);
            completeProgress();
            renderResults(data);
        } catch (error) {
            hideProgress();
            showToast(error.message);
        } finally {
            setLoading(btn, false);
        }
    }

    function setLoading(btn, isLoading) {
        const textSpan = btn.querySelector('.btn-text');
        const spinner = btn.querySelector('.spinner');
        btn.disabled = isLoading;
        if (isLoading) {
            textSpan.classList.add('hidden');
            spinner.classList.remove('hidden');
        } else {
            textSpan.classList.remove('hidden');
            spinner.classList.add('hidden');
        }
    }

    function renderResults(data) {
        document.getElementById('results').classList.remove('hidden');
        document.getElementById('app-layout').classList.add('has-results');

        // Scroll to results
        setTimeout(() => {
            document.getElementById('results').scrollIntoView({ behavior: 'smooth' });
        }, 100);

        const credibility = data.credibility_score; // 0-100 (high = real)

        // ── Decide whether to show FAKE% or REAL% ──
        // If credibility < 50 → the image is more fake → show FAKE %
        // If credibility >= 50 → the image is more real → show REAL %
        const isFake = credibility < 50;
        const displayPct = isFake ? (100 - credibility) : credibility;
        const displayLabel = isFake ? 'FAKE' : 'REAL';

        // ── Color logic ──
        // High fake (≥70% fake) → red
        // High real (≥70% real) → green
        // Everything in between → yellow
        let color, colorRaw;
        if (isFake && displayPct >= 70) {
            color = 'var(--red)';   colorRaw = '#ef4444';
        } else if (!isFake && displayPct >= 70) {
            color = 'var(--green)'; colorRaw = '#10b981';
        } else {
            color = 'var(--amber)'; colorRaw = '#f59e0b';
        }

        // ── Meter ring: fill based on displayPct ──
        const scoreCircle = document.getElementById('score-circle');
        const scoreText   = document.getElementById('score-text');
        const scoreLabel  = document.getElementById('score-label');

        scoreCircle.style.stroke = color;
        scoreText.style.fill     = color;
        scoreLabel.style.fill    = color;
        scoreLabel.textContent   = displayLabel;

        // Animate ring fill
        setTimeout(() => {
            scoreCircle.setAttribute('stroke-dasharray', `${displayPct}, 100`);
        }, 100);

        // Animate number counter
        let current = 0;
        const interval = setInterval(() => {
            if (current >= displayPct) {
                clearInterval(interval);
                scoreText.textContent = `${displayPct}%`;
            } else {
                current++;
                scoreText.textContent = `${current}%`;
            }
        }, 14);

        // ── Meter title: "99% FAKE" or "72% REAL" as the big heading ──
        const meterTitle = document.getElementById('meter-title');
        meterTitle.textContent = `${displayPct}% ${displayLabel}`;
        meterTitle.style.color = colorRaw;

        // ── Trusted boost badge ──
        const existingBadge = document.getElementById('trusted-boost-badge');
        if (existingBadge) existingBadge.remove();
        if (data.trusted_boost_applied) {
            const badge = document.createElement('div');
            badge.id = 'trusted-boost-badge';
            badge.className = 'trusted-boost-badge';
            badge.innerHTML = '⭐ Trusted Source Boost Applied <span>+50%</span>';
            document.querySelector('.meter-card').appendChild(badge);
        }

        // ── Verdict & summary ──
        const verdictEl = document.getElementById('verdict-text');
        verdictEl.textContent = data.verdict;
        verdictEl.style.color = color;

        document.getElementById('summary-text').textContent = data.summary;
        document.getElementById('bias-val').textContent = data.bias;
        document.getElementById('sentiment-val').textContent = data.sentiment;

        // ── Lists ──
        populateList('red-flags-list', data.red_flags || ['None detected']);
        populateList('green-flags-list', data.green_flags || ['None detected']);
        populateList('recommendations-list', data.recommendations || []);
    }

    function populateList(elementId, items) {
        const ul = document.getElementById(elementId);
        ul.innerHTML = '';
        items.forEach(item => {
            const li = document.createElement('li');
            li.textContent = item;
            ul.appendChild(li);
        });
    }
});
