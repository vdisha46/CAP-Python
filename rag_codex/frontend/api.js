const API_BASE = '/api';

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, options);
  if (!response.ok) { const detail = await response.text(); throw new Error(detail || 'Request failed'); }
  return response.json();
}

async function sendQuestion(question) { return request('/chat', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({question}) }); }
async function loadDashboard() { return request('/dashboard'); }
async function loadVendorInvoices() { return request('/vendors'); }
async function loadCustomerInvoices() { return request('/customers'); }
async function loadHistory() { return request('/history'); }
async function uploadFile(file) { const form = new FormData(); form.append('file', file); return request('/upload', { method: 'POST', body: form }); }
async function reindexDocuments() { return request('/reindex', { method: 'POST' }); }
