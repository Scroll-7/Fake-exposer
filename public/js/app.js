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
    const fileNameDisplay = document.getElementById('file-name');
    const analyzeImageBtn = document.getElementById('analyze-image-btn');
    let selectedFile = null;

    dropZone.addEventListener('click', () => fileInput.click());

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
        fileNameDisplay.textContent = `Selected: ${file.name}`;
        analyzeImageBtn.classList.remove('hidden');

        // Show image preview inside the drop zone
        const reader = new FileReader();
        reader.onload = (e) => {
            dropZone.innerHTML = `
                <img src="${e.target.result}" alt="Selected image" id="image-preview"
                     style="max-width:100%; max-height:220px; border-radius:10px; object-fit:contain; display:block; margin:0 auto;">
                <p style="margin-top:10px; font-size:0.8rem; color:var(--text-muted);">Click to change image</p>
                <input type="file" id="file-input" accept="image/*" hidden>
            `;
            // Re-bind file input after re-render
            const newInput = dropZone.querySelector('#file-input');
            dropZone.addEventListener('click', () => newInput.click(), { once: true });
            newInput.addEventListener('change', (e) => {
                if (e.target.files.length) handleFile(e.target.files[0]);
            });
        };
        reader.readAsDataURL(file);
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

        await performAnalysis('/api/analyze/text', { text }, analyzeTextBtn);
    });

    analyzeUrlBtn.addEventListener('click', async () => {
        const url = document.getElementById('url-input').value;
        if (!url.trim() || !url.startsWith('http')) return showToast('Please enter a valid URL.');
        await performAnalysis('/api/analyze/url', { url }, analyzeUrlBtn);
    });

    analyzeImageBtn.addEventListener('click', async () => {
        if (!selectedFile) return showToast('No image selected.');
        
        const formData = new FormData();
        formData.append('image', selectedFile);

        setLoading(analyzeImageBtn, true);
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 60000); // 60s timeout
        try {
            const res = await fetch('/api/analyze/image', {
                method: 'POST',
                body: formData,
                signal: controller.signal
            });
            clearTimeout(timeoutId);
            const data = await res.json();
            if (data.error) throw new Error(data.error);
            renderResults(data);
        } catch (error) {
            clearTimeout(timeoutId);
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
            renderResults(data);
        } catch (error) {
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

        // Score Meter
        const scoreCircle = document.getElementById('score-circle');
        const scoreText = document.getElementById('score-text');
        const score = data.credibility_score;
        
        // Show trusted boost badge if applicable
        const existingBadge = document.getElementById('trusted-boost-badge');
        if (existingBadge) existingBadge.remove();
        if (data.trusted_boost_applied) {
            const badge = document.createElement('div');
            badge.id = 'trusted-boost-badge';
            badge.className = 'trusted-boost-badge';
            badge.innerHTML = '⭐ Trusted Source Boost Applied <span>+50%</span>';
            document.querySelector('.meter-card').appendChild(badge);
        }
        
        // Animate stroke dasharray (0 to score)
        setTimeout(() => {
            scoreCircle.setAttribute('stroke-dasharray', `${score}, 100`);
        }, 100);

        // Color based on score
        let color = 'var(--red)';
        if (score > 40) color = 'var(--amber)';
        if (score > 70) color = 'var(--green)';
        
        scoreCircle.style.stroke = color;
        scoreText.style.fill = color;
        
        // Animate number
        let currentScore = 0;
        const interval = setInterval(() => {
            if (currentScore >= score) {
                clearInterval(interval);
                scoreText.textContent = `${score}%`;
            } else {
                currentScore++;
                scoreText.textContent = `${currentScore}%`;
            }
        }, 15); // Duration depends on score

        // Texts
        const verdictEl = document.getElementById('verdict-text');
        verdictEl.textContent = data.verdict;
        verdictEl.style.color = color;

        document.getElementById('summary-text').textContent = data.summary;
        document.getElementById('bias-val').textContent = data.bias;
        document.getElementById('sentiment-val').textContent = data.sentiment;

        // Lists
        populateList('red-flags-list', data.red_flags || ["None detected"]);
        populateList('green-flags-list', data.green_flags || ["None detected"]);
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
