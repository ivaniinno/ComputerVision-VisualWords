// Tab switching
(function() {
  const tabs = document.querySelectorAll('.tabs ul li');
  tabs.forEach(li => {
    li.addEventListener('click', () => {
      tabs.forEach(x => x.classList.remove('is-active'));
      li.classList.add('is-active');
      const tab = li.getAttribute('data-tab');
      document.querySelectorAll('.tab-pane').forEach(p => p.style.display = 'none');
      document.getElementById(tab).style.display = '';
    });
  });
})();

let currentQueryFilename = null;
let currentUploadId = null;
let simHistChart = null;

async function apiSuggest(q) {
  const r = await fetch(`/api/suggest?query=${encodeURIComponent(q)}`);
  return r.json();
}

async function apiSearchByFilename(filename, topk) {
  const r = await fetch('/api/search/by-filename', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ filename, top_k: topk })
  });
  return r.json();
}

async function apiSearchByUpload(file, topk) {
  const fd = new FormData();
  fd.append('file', file);
  fd.append('top_k', String(topk));
  const r = await fetch('/api/search/by-upload', { method: 'POST', body: fd });
  return r.json();
}

async function apiBinItemsByFilename(qFilename, binIndex, limit=60) {
  const r = await fetch('/api/search/bin-items', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query_filename: qFilename, bin_index: binIndex, limit })
  });
  return r.json();
}

async function apiBinItemsByUpload(uploadId, binIndex, limit=60) {
  const r = await fetch('/api/search/bin-items', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ upload_id: uploadId, bin_index: binIndex, limit })
  });
  return r.json();
}

async function apiDatasetList(params) {
  const qs = new URLSearchParams(params).toString();
  const r = await fetch(`/api/dataset?${qs}`);
  return r.json();
}

async function apiExplain({ qFilename=null, uploadId=null, jFilename, top=15 }) {
  const payload = { candidate_filename: jFilename, top };
  if (qFilename) payload.query_filename = qFilename;
  if (uploadId) payload.upload_id = uploadId;
  const r = await fetch('/api/explain', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });
  return r.json();
}

function renderSuggestions(items) {
  const box = document.getElementById('suggestions');
  box.innerHTML = '';
  if (!items || !items.length) return;
  const ul = document.createElement('ul');
  ul.className = 'menu-list';
  items.forEach(it => {
    const li = document.createElement('li');
    const a = document.createElement('a');
    a.textContent = `${it.filename} ${it.pic_name ? ' — ' + it.pic_name : ''} ${it.author ? ' — ' + it.author : ''}`;
    a.addEventListener('click', () => {
      document.getElementById('search-input').value = it.filename;
      currentQueryFilename = it.filename;
      box.innerHTML = '';
    });
    li.appendChild(a);
    ul.appendChild(li);
  });
  box.appendChild(ul);
}

function simBarHTML(v) {
  const pct = Math.max(0, Math.min(1, v)) * 100;
  return `<div class="sim-bar"><div class="sim-fill" style="width:${pct}%"></div></div>`;
}

function imageHTML(url, alt='') {
  const safeAlt = (alt || '').replace(/"/g, '&quot;');
  return `<div class="image-thumb"><img src="${url}" alt="${safeAlt}" loading="lazy"></div>`;
}

function renderResults(res) {
  const cont = document.getElementById('search-results');
  cont.innerHTML = '';
  if (!res || !res.hits) return;

  const grid = document.createElement('div');
  grid.className = 'columns is-multiline';

  res.hits.forEach(h => {
    const col = document.createElement('div');
    col.className = 'column is-3';
    const imgUrl = h.image_url || (`/archive/${h.filename}`);
    col.innerHTML = `
      <div class="card">
        <div class="card-image">${imageHTML(imgUrl, h.pic_name || h.filename)}</div>
        <div class="card-content">
          <p class="title is-6">${h.pic_name || ''}</p>
          <p class="subtitle is-7">${h.filename}</p>
          <div class="card-meta">${h.author || ''}</div>
          <div class="mt-2">Similarity: <b>${h.similarity.toFixed(3)}</b></div>
          ${simBarHTML(h.similarity)}
          <div class="mt-3">
            <a class="button is-small is-info btn-explain" data-fn="${h.filename}">Explain</a>
          </div>
        </div>
      </div>`;
    grid.appendChild(col);
  });

  cont.appendChild(grid);

  if (res.hist && res.hist.counts && res.hist.edges) {
    const ctx = document.getElementById('sim-hist').getContext('2d');
    const labels = [];
    for (let i = 0; i < res.hist.edges.length - 1; i++) {
      const a = res.hist.edges[i].toFixed(2);
      const b = res.hist.edges[i+1].toFixed(2);
      labels.push(`${a}-${b}`);
    }
    const data = {
      labels,
      datasets: [{ label: 'Cosine similarity', data: res.hist.counts, backgroundColor: '#00d1b2' }]
    };
    if (simHistChart) simHistChart.destroy();
    simHistChart = new Chart(ctx, { type: 'bar', data, options: {
      plugins: { legend: { labels: { color: '#ddd' }}},
      scales: {
        x: { ticks: { color: '#bbb', maxRotation: 0, autoSkip: true }, grid: { color: '#333' }},
        y: { ticks: { color: '#bbb' }, grid: { color: '#333' }}
      },
      onClick: async (evt) => {
        const points = simHistChart.getElementsAtEventForMode(evt, 'nearest', { intersect: true }, true);
        if (!points.length) return;
        const idx = points[0].index;
        await loadBinItems(idx);
      }
    }});
  }

  document.querySelectorAll('.btn-explain').forEach(btn => {
    btn.addEventListener('click', async () => {
      const candidate = btn.getAttribute('data-fn');
      const e = await apiExplain({ qFilename: currentQueryFilename, uploadId: currentUploadId, jFilename: candidate, top: 15 });
      const box = document.getElementById('explain-content');
      if (e.error) {
        box.textContent = e.error;
      } else {
        const rows = e.top_contributions.map(r => `<tr><td>${r.word}</td><td>${r.q.toFixed(4)}</td><td>${r.x.toFixed(4)}</td><td>${r.product.toFixed(6)}</td></tr>`).join('');
        box.innerHTML = `
          <p><b>Query:</b> ${currentQueryFilename || '(uploaded image)'}</p>
          <p><b>Candidate:</b> ${e.pair.filename}</p>
          <p><b>Total similarity:</b> ${e.total_similarity.toFixed(4)}</p>
          <table class="table is-fullwidth is-striped is-narrow">
            <thead><tr><th>Word #</th><th>q</th><th>x</th><th>q*x</th></tr></thead>
            <tbody>${rows}</tbody>
          </table>`;
      }
      document.getElementById('explain-modal').classList.add('is-active');
    });
  });
}

function renderBinItemsList(items, rangeLabel) {
  const cont = document.getElementById('bin-items');
  cont.innerHTML = '';
  const title = document.createElement('h3');
  title.className = 'title is-6';
  title.textContent = `Items in selected bin ${rangeLabel} (top shown)`;
  cont.appendChild(title);
  const grid = document.createElement('div');
  grid.className = 'columns is-multiline';
  (items || []).forEach(h => {
    const col = document.createElement('div');
    col.className = 'column is-3';
    const imgUrl = h.image_url || (`/archive/${h.filename}`);
    col.innerHTML = `
      <div class="card">
        <div class="card-image">${imageHTML(imgUrl, h.pic_name || h.filename)}</div>
        <div class="card-content">
          <p class="title is-6">${h.pic_name || ''}</p>
          <p class="subtitle is-7">${h.filename}</p>
          <div class="card-meta">${h.author || ''}</div>
          <div class="mt-2">Similarity: <b>${h.similarity.toFixed(3)}</b></div>
          ${simBarHTML(h.similarity)}
          <div class="mt-3"><a class="button is-small is-info btn-explain-bin" data-fn="${h.filename}">Explain</a></div>
        </div>
      </div>`;
    grid.appendChild(col);
  });
  cont.appendChild(grid);

  grid.querySelectorAll('.btn-explain-bin').forEach(btn => {
    btn.addEventListener('click', async () => {
      const candidate = btn.getAttribute('data-fn');
      const e = await apiExplain({ qFilename: currentQueryFilename, uploadId: currentUploadId, jFilename: candidate, top: 15 });
      const box = document.getElementById('explain-content');
      if (e.error) {
        box.textContent = e.error;
      } else {
        const rows = e.top_contributions.map(r => `<tr><td>${r.word}</td><td>${r.q.toFixed(4)}</td><td>${r.x.toFixed(4)}</td><td>${r.product.toFixed(6)}</td></tr>`).join('');
        box.innerHTML = `
          <p><b>Query:</b> ${currentQueryFilename || '(uploaded image)'}</p>
          <p><b>Candidate:</b> ${e.pair.filename}</p>
          <p><b>Total similarity:</b> ${e.total_similarity.toFixed(4)}</p>
          <table class="table is-fullwidth is-striped is-narrow">
            <thead><tr><th>Word #</th><th>q</th><th>x</th><th>q*x</th></tr></thead>
            <tbody>${rows}</tbody>
          </table>`;
      }
      document.getElementById('explain-modal').classList.add('is-active');
    });
  });
}

async function loadBinItems(binIndex) {
  let res;
  if (currentUploadId) {
    res = await apiBinItemsByUpload(currentUploadId, binIndex, 60);
  } else if (currentQueryFilename) {
    res = await apiBinItemsByFilename(currentQueryFilename, binIndex, 60);
  } else {
    return;
  }
  const rangeLabel = res.range ? `[${res.range[0].toFixed(3)}, ${res.range[1].toFixed(3)})` : '';
  renderBinItemsList(res.items || [], rangeLabel);
}

// Search UI events
(function() {
  const btnSuggest = document.getElementById('btn-suggest');
  const btnSearch = document.getElementById('btn-search');
  const input = document.getElementById('search-input');

  btnSuggest.addEventListener('click', async () => {
    const q = input.value.trim();
    const res = await apiSuggest(q);
    renderSuggestions(res.items || []);
  });

  btnSearch.addEventListener('click', async () => {
    let filename = input.value.trim();
    if (!filename && currentQueryFilename) filename = currentQueryFilename;
    if (!filename) return alert('Please select a filename via Suggest or type exact');
    currentQueryFilename = filename;
    currentUploadId = null;
    const topk = document.getElementById('topk-select').value;
    const res = await apiSearchByFilename(filename, parseInt(topk, 10));
    if (res.error) { alert(res.error); return; }
    renderResults(res);
  });
})();

// Browse list
(function() {
  const authorSel = document.getElementById('filter-author');
  const btnApply = document.getElementById('btn-apply');
  const inputQ = document.getElementById('browse-query');
  const browseCont = document.getElementById('browse-list');
  const genresPicked = new Set();

  document.querySelectorAll('.genre-toggle').forEach(x => {
    x.addEventListener('click', () => {
      const g = x.getAttribute('data-genre');
      if (genresPicked.has(g)) { genresPicked.delete(g); x.textContent = 'add'; x.classList.remove('is-danger'); x.classList.add('is-info'); }
      else { genresPicked.add(g); x.textContent = 'picked'; x.classList.remove('is-info'); x.classList.add('is-danger'); }
    });
  });

  async function loadPage(page=1) {
    const params = {
      page,
      size: 24,
      author: authorSel.value,
      query: inputQ.value.trim(),
    };
    if (genresPicked.size) params.genres = Array.from(genresPicked).join(',');
    const r = await apiDatasetList(params);
    browseCont.innerHTML = '';
    const grid = document.createElement('div');
    grid.className = 'columns is-multiline';
    r.items.forEach(item => {
      const col = document.createElement('div');
      col.className = 'column is-3';
      const imgUrl = item.image_url || (`/archive/${item.filename}`);
      col.innerHTML = `
        <div class="card">
          <div class="card-image">${imageHTML(imgUrl, item.pic_name || item.filename)}</div>
          <div class="card-content">
            <p class="title is-6">${item.pic_name || ''}</p>
            <p class="subtitle is-7">${item.filename}</p>
            <div class="card-meta">${item.author || ''}</div>
            <div class="mt-3"><a class="button is-small is-primary btn-find" data-fn="${item.filename}">Find similar</a></div>
          </div>
        </div>`;
      grid.appendChild(col);
    });

    const footer = document.createElement('div');
    footer.className = 'mt-4';
    const pages = Math.ceil(r.total / r.size);
    const pag = document.createElement('nav');
    pag.className = 'pagination';
    const ul = document.createElement('ul');
    ul.className = 'pagination-list';
    for (let p = 1; p <= Math.min(pages, 7); p++) {
      const li = document.createElement('li');
      const a = document.createElement('a');
      a.className = 'pagination-link' + (p === r.page ? ' is-current' : '');
      a.textContent = p;
      a.addEventListener('click', () => loadPage(p));
      li.appendChild(a);
      ul.appendChild(li);
    }
    pag.appendChild(ul);
    footer.appendChild(pag);

    browseCont.appendChild(grid);
    browseCont.appendChild(footer);

    grid.querySelectorAll('.btn-find').forEach(btn => {
      btn.addEventListener('click', async () => {
        currentQueryFilename = btn.getAttribute('data-fn');
        const res = await apiSearchByFilename(currentQueryFilename, 12);
        document.querySelector('[data-tab="tab-search"]').click();
        document.getElementById('search-input').value = currentQueryFilename;
        renderResults(res);
      });
    });
  }

  btnApply.addEventListener('click', () => loadPage(1));
  loadPage(1);
})();

// Modal close
(function() {
  const m = document.getElementById('explain-modal');
  document.getElementById('explain-close').addEventListener('click', () => m.classList.remove('is-active'));
  document.getElementById('explain-close2').addEventListener('click', () => m.classList.remove('is-active'));
  m.querySelector('.modal-background').addEventListener('click', () => m.classList.remove('is-active'));
})();

// Upload search
(function() {
  const btn = document.getElementById('btn-upload-search');
  const fileInput = document.getElementById('upload-file');
  btn.addEventListener('click', async () => {
    const f = fileInput.files && fileInput.files[0];
    if (!f) { alert('Choose an image file'); return; }
    const topk = parseInt(document.getElementById('upload-topk-select').value, 10);
    const res = await apiSearchByUpload(f, topk);
    if (res.error) { alert(res.error); return; }
    currentUploadId = res.upload_id || null;
    currentQueryFilename = null;
    renderResults(res);
    window.scrollTo({ top: 0, behavior: 'smooth' });
    document.getElementById('search-input').value = '';
  });
})();
