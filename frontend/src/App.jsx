import { useState, useMemo, useEffect, useCallback } from 'react';
import axios from 'axios';
import {
  PieChart, Pie, Cell, ResponsiveContainer, Tooltip,
  BarChart, Bar, XAxis, YAxis, CartesianGrid
} from 'recharts';
import Login from './Login';

const API = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

// Auto-logout on token expiry
axios.interceptors.response.use(
  r => r,
  err => {
    if (err.response?.status === 401) {
      localStorage.removeItem('token');
      window.location.reload();
    }
    return Promise.reject(err);
  }
);

function App() {
  const [token, setToken]               = useState(localStorage.getItem('token'));
  const [file, setFile]                 = useState(null);
  const [results, setResults]           = useState(null);
  const [loading, setLoading]           = useState(false);
  const [searchTerm, setSearchTerm]     = useState('');
  const [activeTab, setActiveTab]       = useState('dashboard');
  const [selectedStudent, setSelectedStudent] = useState(null);
  const [isDrafting, setIsDrafting]     = useState(false);
  const [aiDraft, setAiDraft]           = useState(null);
  const [scrollTop, setScrollTop]       = useState(0);
  const [history, setHistory]           = useState([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [modelAccuracy, setModelAccuracy]   = useState(null);

  useEffect(() => { setAiDraft(null); }, [selectedStudent]);

  // Fetch model accuracy from /health on mount
  useEffect(() => {
    if (!token) return;
    axios.get(`${API}/health`, { headers: { Authorization: `Bearer ${token}` } })
      .then(r => { if (r.data.model_accuracy) setModelAccuracy((r.data.model_accuracy * 100).toFixed(1)); })
      .catch(() => {});
  }, [token]);

  // Fetch prediction history when History tab is opened
  useEffect(() => {
    if (activeTab !== 'history' || !token) return;
    setHistoryLoading(true);
    axios.get(`${API}/predict/history`, { headers: { Authorization: `Bearer ${token}` } })
      .then(r => setHistory(r.data))
      .catch(() => {})
      .finally(() => setHistoryLoading(false));
  }, [activeTab, token]);

  const chartData = useMemo(() => {
    if (!results) return [];
    return [
      { name: 'Stable',  value: results.total_students - results.at_risk_count, color: '#3FD18B' },
      { name: 'At Risk', value: results.at_risk_count, color: '#FF4A3D' }
    ];
  }, [results]);

  if (!token) return <Login setToken={setToken} />;

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);
    try {
      const response = await axios.post(`${API}/predict/batch`, formData, {
        headers: { Authorization: `Bearer ${token}` }
      });
      const d = response.data;
      d.at_risk_count = d.data.filter(s => s.is_at_risk).length;
      setResults(d);
      setActiveTab('dashboard');
    } catch (err) {
      alert('Upload failed. Is the API running?');
    } finally {
      setLoading(false);
    }
  };

  // Fixed streaming: parse SSE "data: ..." lines properly
  const handleGenerateDraft = async () => {
    if (!selectedStudent) return;
    setIsDrafting(true);
    setAiDraft('');
    try {
      const response = await axios.post(`${API}/agent/draft-intervention`, {
        student_id:   selectedStudent.student_id,
        risk_prob:    selectedStudent.risk_probability,
        risk_factors: selectedStudent.risk_factors
      }, {
        headers: { Authorization: `Bearer ${token}` }
      });
      setAiDraft(response.data.plan);
    } catch (err) {
      setAiDraft('⚠ AI generation failed. Check API key and backend logs.');
    } finally {
      setIsDrafting(false);
    }
  };


  const handleAction = async (studentId, currentlyActioned) => {
    const endpoint = currentlyActioned ? `${API}/predict/unaction/${studentId}` : `${API}/predict/action/${studentId}`;
    try {
      await axios.post(endpoint, {}, { headers: { Authorization: `Bearer ${token}` } });
      setResults(prev => {
        if (!prev) return prev;
        return {
          ...prev,
          data: prev.data.map(s =>
            s.student_id === studentId ? { ...s, actioned: !currentlyActioned } : s
          )
        };
      });
      if (selectedStudent?.student_id === studentId) {
        setSelectedStudent(prev => ({ ...prev, actioned: !currentlyActioned }));
      }
    } catch (err) {
      alert('Failed to update action status.');
    }
  };

  const rowHeight = 54;
  const containerHeight = 500;
  const filteredData = results
    ? results.data.filter(s => s.student_id.toLowerCase().includes(searchTerm.toLowerCase()))
    : [];
  const startIndex  = Math.max(0, Math.floor(scrollTop / rowHeight) - 2);
  const endIndex    = Math.min(filteredData.length, startIndex + Math.ceil(containerHeight / rowHeight) + 4);
  const visibleData = filteredData.slice(startIndex, endIndex);

  // Group history by batch for the history tab
  const historyByBatch = useMemo(() => {
    const map = {};
    history.forEach(r => {
      if (!map[r.batch_id]) map[r.batch_id] = { batch_id: r.batch_id, records: [], at_risk: 0 };
      map[r.batch_id].records.push(r);
      if (r.is_at_risk) map[r.batch_id].at_risk++;
    });
    return Object.values(map).reverse();
  }, [history]);

  return (
    <div className="flex min-h-screen bg-[#08080A] text-[#EDEDE7] font-['IBM_Plex_Sans'] relative overflow-hidden">

      {/* Background gradient */}
      <div className="fixed inset-0 pointer-events-none z-0 opacity-50 bg-[radial-gradient(120%_90%_at_80%_-10%,rgba(255,176,0,.04),transparent_55%),radial-gradient(90%_70%_at_-10%_110%,rgba(91,168,255,.03),transparent_50%)]"></div>

      {/* ── Student Detail Modal ─────────────────────────────────────── */}
      {selectedStudent && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
          <div className="bg-[#0F0F12] border border-[#242429] w-full max-w-4xl shadow-2xl overflow-hidden flex flex-col max-h-[90vh]">
            <div className="p-6 border-b border-[#242429] flex justify-between items-start bg-[#08080A]">
              <div>
                <div className="font-['IBM_Plex_Mono'] text-[10px] tracking-[2.5px] text-[#FFB000] uppercase mb-2">
                  Explainable AI · Roll {selectedStudent.student_id}
                </div>
                <h1 className="font-['Saira_Stencil_One'] text-3xl tracking-wide">Why Flagged?</h1>
                <p className="text-[#9A9AA1] text-sm mt-2 font-['IBM_Plex_Sans'] max-w-xl">
                  SHAP values decompose the prediction into per-feature contributions. Red pushes risk up, green pulls it down.
                </p>
              </div>
              <div className="flex items-center gap-3">
                {selectedStudent.is_at_risk && (
                  <button
                    onClick={() => handleAction(selectedStudent.student_id, selectedStudent.actioned)}
                    className={`font-['Oswald'] text-[11px] tracking-[1px] uppercase px-4 py-2 border transition-colors ${selectedStudent.actioned ? 'border-[#3FD18B] text-[#3FD18B] bg-[rgba(63,209,139,.08)] hover:bg-[rgba(63,209,139,.15)]' : 'border-[#FFB000] text-[#FFB000] bg-[rgba(255,176,0,.08)] hover:bg-[rgba(255,176,0,.15)]'}`}
                  >
                    {selectedStudent.actioned ? '✓ Actioned' : 'Mark as Actioned'}
                  </button>
                )}
                <button onClick={() => setSelectedStudent(null)} className="text-[#62626B] hover:text-[#EDEDE7] transition p-2 text-xl font-bold">✕</button>
              </div>
            </div>

            <div className="p-6 flex-1 overflow-y-auto grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* SHAP chart */}
              <div className="bg-[#0F0F12] border border-[#242429] p-5">
                <h3 className="font-['Oswald'] text-[12px] tracking-[1.8px] uppercase text-[#9A9AA1] mb-6 flex items-center gap-2">
                  <svg className="w-4 h-4 stroke-[#62626B] fill-none" viewBox="0 0 24 24"><polygon points="12 2 22 8.5 12 15 2 8.5"/><polyline points="2 15.5 12 22 22 15.5"/></svg>
                  Feature Contribution (SHAP)
                </h3>
                <div className="h-64 w-full">
                  {selectedStudent.risk_factors?.length > 0 ? (
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart
                        data={selectedStudent.risk_factors}
                        layout="vertical"
                        margin={{ top: 5, right: 30, left: 50, bottom: 5 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" stroke="#242429" horizontal={false} />
                        <XAxis type="number" stroke="#62626B" tick={{ fill: '#62626B', fontSize: 10, fontFamily: 'IBM Plex Mono' }} />
                        <YAxis dataKey="feature" type="category" width={90} stroke="#62626B" tick={{ fill: '#9A9AA1', fontSize: 11, fontFamily: 'IBM Plex Mono' }} />
                        <Tooltip
                          contentStyle={{ backgroundColor: '#0F0F12', border: '1px solid #242429', fontFamily: 'IBM Plex Mono', fontSize: '11px' }}
                          itemStyle={{ color: '#FF4A3D' }}
                          cursor={{ fill: '#15151A' }}
                        />
                        <Bar dataKey="impact" barSize={16}>
                          {selectedStudent.risk_factors.map((entry, i) => (
                            <Cell key={i} fill={entry.impact >= 0 ? '#FF4A3D' : '#3FD18B'} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="h-full flex items-center justify-center text-sm text-[#62626B] border border-dashed border-[#34343C]">
                      No significant risk drivers identified.
                    </div>
                  )}
                </div>
              </div>

              {/* Intervention plan panel */}
              <div className="bg-[#0F0F12] border border-[#242429] p-5 flex flex-col">
                <div className="flex justify-between items-center mb-6">
                  <h3 className="font-['Oswald'] text-[12px] tracking-[1.8px] uppercase text-[#9A9AA1] flex items-center gap-2">
                    <svg className="w-4 h-4 stroke-[#62626B] fill-none" viewBox="0 0 24 24"><rect x="6" y="4" width="12" height="17" rx="1"/><path d="M9 4V3h6v1"/><line x1="9" y1="9" x2="15" y2="9"/><line x1="9" y1="13" x2="15" y2="13"/></svg>
                    Intervention Plan
                  </h3>
                  {selectedStudent.is_at_risk && (
                    <button
                      onClick={handleGenerateDraft}
                      disabled={isDrafting}
                      className="font-['Oswald'] font-medium text-[12px] tracking-[1px] uppercase px-4 py-2 border border-[#EDEDE7] bg-[#EDEDE7] text-black hover:bg-white transition-colors disabled:opacity-50"
                    >
                      {isDrafting ? 'Consulting AI...' : 'Generate Custom Plan'}
                    </button>
                  )}
                </div>
                <div className="flex-1 bg-[#15151A] p-5 border border-[#34343C] text-[13px] leading-relaxed text-[#9A9AA1] overflow-y-auto min-h-[200px] max-h-[350px]">
                  {isDrafting && !aiDraft ? (
                    <div className="flex items-center justify-center h-full text-[#62626B] animate-pulse font-['IBM_Plex_Mono'] text-[11px] uppercase tracking-widest">
                      Agent generating protocol...
                    </div>
                  ) : aiDraft ? (
                    <div className="text-[#EDEDE7] whitespace-pre-wrap text-[13px] leading-relaxed">
                      {aiDraft.split('\n').map((line, i) => {
                        const bold = line.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
                        return <p key={i} className={line.startsWith('*') ? 'ml-3 my-1' : 'my-1'} dangerouslySetInnerHTML={{__html: bold}} />;
                      })}
                    </div>
                  ) : (
                    <div className="h-full flex flex-col items-center justify-center text-center gap-3">
                      {selectedStudent.is_at_risk ? (
                        <>
                          <svg className="w-8 h-8 stroke-[#FFB000] fill-none stroke-[1.4]" viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><line x1="12" y1="8" x2="12" y2="12"/><circle cx="12" cy="16" r=".6"/></svg>
                          <p className="text-[#9A9AA1] text-[12px] font-['IBM_Plex_Mono']">Click <span className="text-[#EDEDE7]">Generate Custom Plan</span> to get an AI-powered intervention plan for this student.</p>
                        </>
                      ) : (
                        <>
                          <svg className="w-8 h-8 stroke-[#3FD18B] fill-none stroke-[1.4]" viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><polyline points="9 12 11 14 15 10"/></svg>
                          <p className="text-[#62626B] text-[12px] font-['IBM_Plex_Mono']">Student is within stable thresholds. No intervention required.</p>
                        </>
                      )}
                    </div>
                  )}
                </div>
              </div>

              {/* Student raw stats */}
              <div className="lg:col-span-2 bg-[#0F0F12] border border-[#242429] p-5">
                <h3 className="font-['Oswald'] text-[12px] tracking-[1.8px] uppercase text-[#9A9AA1] mb-4">Risk Summary</h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="bg-[#15151A] border border-[#242429] p-4">
                    <div className="font-['IBM_Plex_Mono'] text-[10px] text-[#62626B] uppercase tracking-wider mb-1">Risk Score</div>
                    <div className="font-['Saira_Stencil_One'] text-2xl" style={{ color: selectedStudent.is_at_risk ? '#FF4A3D' : '#3FD18B' }}>
                      {selectedStudent.risk_probability}%
                    </div>
                  </div>
                  <div className="bg-[#15151A] border border-[#242429] p-4">
                    <div className="font-['IBM_Plex_Mono'] text-[10px] text-[#62626B] uppercase tracking-wider mb-1">Status</div>
                    <div className={`font-['Oswald'] text-xl tracking-wide ${selectedStudent.is_at_risk ? 'text-[#FF4A3D]' : 'text-[#3FD18B]'}`}>
                      {selectedStudent.is_at_risk ? 'HIGH RISK' : 'ON TRACK'}
                    </div>
                  </div>
                  <div className="bg-[#15151A] border border-[#242429] p-4">
                    <div className="font-['IBM_Plex_Mono'] text-[10px] text-[#62626B] uppercase tracking-wider mb-1">Top Driver</div>
                    <div className="font-['IBM_Plex_Mono'] text-sm text-[#EDEDE7] truncate">
                      {selectedStudent.risk_factors?.[0]?.feature || '—'}
                    </div>
                  </div>
                  <div className="bg-[#15151A] border border-[#242429] p-4">
                    <div className="font-['IBM_Plex_Mono'] text-[10px] text-[#62626B] uppercase tracking-wider mb-1">Plan Ready</div>
                    <div className={`font-['Oswald'] text-xl tracking-wide ${selectedStudent.actioned ? 'text-[#3FD18B]' : 'text-[#FFB000]'}`}>
                      {selectedStudent.actioned ? 'ACTIONED' : (aiDraft || selectedStudent.intervention_plan ? 'READY' : 'PENDING')}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Sidebar ──────────────────────────────────────────────────── */}
      <aside className="w-[200px] min-w-[200px] bg-[#0F0F12] border-r border-[#242429] flex flex-col relative z-20">
        <div className="flex items-center gap-3 p-5 border-b border-[#242429]">
          <svg className="w-[26px] h-[26px] fill-[#EDEDE7]" viewBox="0 0 100 100"><path d="M30 14 H74 V30 H46 V46 H66 V60 H46 V86 H30 Z"/></svg>
          <div>
            <div className="font-['Saira_Stencil_One'] text-[20px] tracking-[1px] leading-none">FAILSAFE</div>
            <div className="font-['IBM_Plex_Mono'] text-[9px] text-[#62626B] tracking-[1.5px] mt-1">Early-Warning System</div>
          </div>
        </div>

        <nav className="flex-1 p-3 overflow-y-auto">
          <div className="font-['Oswald'] uppercase text-[10px] tracking-[1.5px] text-[#62626B] px-3 pt-4 pb-1">Operate</div>
          {[
            { id: 'dashboard', label: 'Dashboard', icon: <><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/></> },
          ].map(t => (
            <button key={t.id} onClick={() => setActiveTab(t.id)}
              className={`w-full flex items-center gap-3 px-3 py-2 font-['Oswald'] text-[13.5px] tracking-[0.4px] transition-colors border-l-2 ${activeTab === t.id ? 'text-[#EDEDE7] border-[#FFB000] bg-[#15151A]' : 'text-[#9A9AA1] border-transparent hover:text-[#EDEDE7] hover:bg-[#15151A]'}`}>
              <svg className="w-[17px] h-[17px] stroke-current fill-none stroke-[1.6]" viewBox="0 0 24 24">{t.icon}</svg>
              {t.label}
            </button>
          ))}

          <div className="font-['Oswald'] uppercase text-[10px] tracking-[1.5px] text-[#62626B] px-3 pt-6 pb-1">Investigate</div>
          {[
            { id: 'analytics', label: 'Analytics', icon: <><line x1="4" y1="20" x2="20" y2="20"/><polyline points="5 16 9 11 13 14 19 6"/></> },
            { id: 'history',   label: 'Batch History', icon: <><circle cx="12" cy="12" r="9"/><polyline points="12 7 12 12 15 15"/></> },
          ].map(t => (
            <button key={t.id} onClick={() => setActiveTab(t.id)}
              className={`w-full flex items-center gap-3 px-3 py-2 font-['Oswald'] text-[13.5px] tracking-[0.4px] transition-colors border-l-2 ${activeTab === t.id ? 'text-[#EDEDE7] border-[#FFB000] bg-[#15151A]' : 'text-[#9A9AA1] border-transparent hover:text-[#EDEDE7] hover:bg-[#15151A]'}`}>
              <svg className="w-[17px] h-[17px] stroke-current fill-none stroke-[1.6]" viewBox="0 0 24 24">{t.icon}</svg>
              {t.label}
            </button>
          ))}
        </nav>

        <div className="p-4 border-t border-[#242429] flex items-center gap-3">
          <div className="w-[30px] h-[30px] rounded-full bg-[#1B1B21] border border-[#34343C] flex items-center justify-center font-['IBM_Plex_Mono'] text-[11px] text-[#9A9AA1]">FA</div>
          <div className="leading-tight min-w-0">
            <b className="font-['IBM_Plex_Sans'] font-semibold text-[12px] block truncate">Faculty Admin</b>
            <span onClick={() => { localStorage.removeItem('token'); setToken(null); }}
              className="font-['IBM_Plex_Mono'] text-[9.5px] text-[#62626B] cursor-pointer hover:text-[#FF4A3D]">Sign Out</span>
          </div>
        </div>
      </aside>

      {/* ── Main content ─────────────────────────────────────────────── */}
      <main className="flex-1 min-w-0 overflow-y-auto relative z-10 flex flex-col">
        <div className="sticky top-0 z-50 flex items-center gap-4 px-7 h-[58px] bg-[#08080A]/80 backdrop-blur-md border-b border-[#242429]">
          <div className="font-['IBM_Plex_Mono'] text-[11px] text-[#62626B] tracking-[0.5px]">
            FAILSAFE / <b className="text-[#9A9AA1] font-normal">
              {{ dashboard: 'Dashboard', analytics: 'Analytics', history: 'Batch History' }[activeTab]}
            </b>
          </div>
          <div className="ml-auto flex items-center gap-2 font-['IBM_Plex_Mono'] text-[10px] text-[#9A9AA1] tracking-[1px]">
            <span className="w-[7px] h-[7px] rounded-full bg-[#3FD18B] animate-pulse"></span>
            MODEL LIVE
          </div>
        </div>

        <div className="p-7">

          {/* ── Dashboard Tab ── */}
          {activeTab === 'dashboard' && (
            <div className="animate-in fade-in slide-in-from-bottom-4 duration-500 max-w-6xl">
              <div className="mb-8 border-b border-[#242429] pb-8">
                <div className="font-['IBM_Plex_Mono'] text-[10px] tracking-[2.5px] text-[#FFB000] uppercase mb-3">Command Center</div>
                <h1 className="font-['Saira_Stencil_One'] text-4xl tracking-wide mb-3">FAILSAFE</h1>
                <p className="text-[#9A9AA1] text-[13px] max-w-3xl leading-[1.6]">
                  Upload your student dataset to identify at-risk students early, use Explainable AI to understand root causes, and automatically generate personalised intervention plans before it's too late.
                </p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                <div className="bg-[#0F0F12] border border-[#242429] p-5 relative group hover:border-[#34343C] transition-colors">
                  <span className="absolute top-0 right-0 w-2.5 h-2.5 border-t border-r border-[#46464F]"></span>
                  <div className="font-['Oswald'] uppercase text-[11px] tracking-[1.8px] text-[#9A9AA1] mb-3">Dataset Input</div>
                  <h3 className="font-['Saira_Stencil_One'] text-[22px] text-[#EDEDE7] truncate mb-2">{file ? file.name : 'Awaiting Data'}</h3>
                  <input type="file" accept=".csv" onChange={e => setFile(e.target.files[0])}
                    className="mt-2 text-[10px] text-[#62626B] font-['IBM_Plex_Mono'] file:mr-3 file:py-1 file:px-3 file:border file:border-[#34343C] file:text-[10px] file:font-['IBM_Plex_Mono'] file:bg-[#0F0F12] file:text-[#9A9AA1] hover:file:bg-[#15151A] hover:file:text-[#EDEDE7] cursor-pointer" />
                </div>

                <div className="bg-[#0F0F12] border border-[#242429] p-5 relative">
                  <span className="absolute top-0 right-0 w-2.5 h-2.5 border-t border-r border-[#46464F]"></span>
                  <div className="font-['Oswald'] uppercase text-[11px] tracking-[1.8px] text-[#9A9AA1]">Risk Flagged</div>
                  <div className="font-['Saira_Stencil_One'] text-[42px] leading-none my-3 text-[#FF4A3D]">{results ? results.at_risk_count : '--'}</div>
                  <div className="font-['IBM_Plex_Mono'] text-[11px] text-[#62626B]">Out of {results ? results.total_students : '0'} Total</div>
                </div>

                <div className="bg-[#0F0F12] border border-[#242429] p-5 flex flex-col justify-center">
                  <button onClick={handleUpload} disabled={loading || !file}
                    className="font-['Oswald'] font-medium text-[12.5px] tracking-[1px] uppercase py-3 border border-[#EDEDE7] bg-[#EDEDE7] text-black hover:bg-white transition-colors disabled:opacity-40 flex items-center justify-center gap-2">
                    {loading ? 'Processing...' : 'Initialize Pipeline'}
                  </button>
                  {results && (
                    <div className="mt-3 font-['IBM_Plex_Mono'] text-[10px] text-[#62626B] text-center">
                      Batch: {results.batch_id} · Threshold: {results.threshold_used}
                    </div>
                  )}
                </div>
              </div>

              {results && (
                <div className="bg-[#0F0F12] border border-[#242429]">
                  <div className="p-4 flex justify-between items-center border-b border-[#242429]">
                    <h3 className="font-['Oswald'] text-[12px] tracking-[1.8px] text-[#9A9AA1] uppercase flex items-center gap-2">
                      <svg className="w-[15px] h-[15px] stroke-[#62626B] fill-none" viewBox="0 0 24 24"><line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/><circle cx="3.5" cy="6" r="1"/><circle cx="3.5" cy="12" r="1"/><circle cx="3.5" cy="18" r="1"/></svg>
                      At-Risk Roster (Batch: {results.batch_id})
                    </h3>
                    <div className="flex items-center gap-2 bg-[#08080A] border border-[#242429] px-3 py-1.5 w-64 focus-within:border-[#FFB000] transition-colors">
                      <svg className="w-[14px] h-[14px] stroke-[#62626B] fill-none stroke-[1.6]" viewBox="0 0 24 24"><circle cx="11" cy="11" r="7"/><line x1="21" y1="21" x2="16.5" y2="16.5"/></svg>
                      <input type="text" placeholder="Search Subject ID..." onChange={e => setSearchTerm(e.target.value)}
                        className="bg-transparent border-none outline-none text-[#EDEDE7] font-['IBM_Plex_Mono'] text-[11px] w-full placeholder-[#62626B]" />
                    </div>
                  </div>

                  <div className="overflow-auto max-h-[500px]" onScroll={e => setScrollTop(e.target.scrollTop)}>
                    <table className="w-full text-left border-collapse">
                      <thead className="sticky top-0 z-10 bg-[#0F0F12]">
                        <tr>
                          {['Roll No', 'Failure Probability', 'Band', 'Top Factors', 'Status', ''].map(h => (
                            <th key={h} className="font-['Oswald'] font-medium uppercase text-[10.5px] tracking-[1.5px] text-[#62626B] px-4 pb-3 pt-3 border-b border-[#242429]">{h}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody className="text-[13px] text-[#9A9AA1]">
                        {startIndex > 0 && <tr style={{ height: `${startIndex * rowHeight}px` }}><td colSpan="5"></td></tr>}
                        {visibleData.map(s => (
                          <tr key={s.student_id} onClick={() => setSelectedStudent(s)}
                            className="hover:bg-[#15151A] transition-colors cursor-pointer group">
                            <td className="px-4 py-3 border-b border-[#242429] font-['IBM_Plex_Mono'] text-[11px] text-[#62626B] group-hover:text-[#EDEDE7]">{s.student_id}</td>
                            <td className="px-4 py-3 border-b border-[#242429]">
                              <div className="flex items-center gap-2">
                                <div className="flex-1 h-[5px] bg-[#1B1B21] relative overflow-hidden">
                                  <i className="absolute left-0 top-0 h-full transition-all" style={{ width: `${s.risk_probability}%`, background: s.is_at_risk ? '#FF4A3D' : '#3FD18B' }}></i>
                                </div>
                                <span className="font-['IBM_Plex_Mono'] text-[12px] w-[40px] text-right" style={{ color: s.is_at_risk ? '#FF4A3D' : '#3FD18B' }}>
                                  {s.risk_probability.toFixed(1)}%
                                </span>
                              </div>
                            </td>
                            <td className="px-4 py-3 border-b border-[#242429]">
                              <span className={`font-['Oswald'] font-medium uppercase text-[10px] tracking-[1px] px-2 py-1 border inline-flex items-center gap-1.5 ${s.is_at_risk ? 'text-[#FF4A3D] border-[rgba(255,74,61,.4)] bg-[rgba(255,74,61,.08)]' : 'text-[#3FD18B] border-[rgba(63,209,139,.4)] bg-[rgba(63,209,139,.08)]'}`}>
                                <span className="w-[6px] h-[6px] rounded-full bg-current"></span>
                                {s.is_at_risk ? 'High' : 'Track'}
                              </span>
                            </td>
                            <td className="px-4 py-3 border-b border-[#242429]">
                              <div className="flex flex-wrap gap-1.5">
                                {s.risk_factors?.slice(0, 3).map((f, idx) => (
                                  <span key={idx} className="font-['Oswald'] text-[9px] tracking-[1px] uppercase px-1.5 border border-[#34343C] text-[#9A9AA1]">
                                    {f.feature}
                                  </span>
                                )) ?? <span className="text-[10px] text-[#62626B]">—</span>}
                              </div>
                            </td>
                            <td className="px-4 py-3 border-b border-[#242429]">
                              {s.is_at_risk && (
                                <span className={`font-['Oswald'] text-[9px] tracking-[1px] uppercase px-2 py-1 border ${s.actioned ? 'text-[#3FD18B] border-[rgba(63,209,139,.4)] bg-[rgba(63,209,139,.08)]' : 'text-[#FFB000] border-[rgba(255,176,0,.4)] bg-[rgba(255,176,0,.08)]'}`}>
                                  {s.actioned ? '✓ Done' : 'Pending'}
                                </span>
                              )}
                            </td>
                            <td className="px-4 py-3 border-b border-[#242429] text-right">
                              <svg viewBox="0 0 24 24" className="w-[15px] h-[15px] stroke-[#62626B] stroke-[1.6] fill-none inline-block"><polyline points="9 6 15 12 9 18"/></svg>
                            </td>
                          </tr>
                        ))}
                        {endIndex < filteredData.length && <tr style={{ height: `${(filteredData.length - endIndex) * rowHeight}px` }}><td colSpan="5"></td></tr>}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* ── Analytics Tab ── */}
          {activeTab === 'analytics' && (
            <div className="animate-in fade-in slide-in-from-bottom-4 duration-500 max-w-6xl">
              <div className="mb-6">
                <div className="font-['IBM_Plex_Mono'] text-[10px] tracking-[2.5px] text-[#FFB000] uppercase mb-2">Cohort Analytics</div>
                <h1 className="font-['Saira_Stencil_One'] text-4xl text-[#EDEDE7] tracking-wide mb-2">Risk Trends</h1>
              </div>

              {results ? (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="bg-[#0F0F12] border border-[#242429] p-5 relative">
                    <span className="absolute top-0 right-0 w-2.5 h-2.5 border-t border-r border-[#46464F]"></span>
                    <h3 className="font-['Oswald'] text-[12px] tracking-[1.8px] text-[#9A9AA1] uppercase flex items-center gap-2 mb-4">
                      <svg className="w-[15px] h-[15px] stroke-[#62626B] fill-none" viewBox="0 0 24 24"><path d="M21 12A9 9 0 1112 3v9z"/><path d="M12 3a9 9 0 019 9h-9z"/></svg>
                      By Risk Band
                    </h3>
                    <div className="h-64 w-full relative">
                      <ResponsiveContainer width="100%" height="100%">
                        <PieChart>
                          <Pie data={chartData} innerRadius={70} outerRadius={90} paddingAngle={2} dataKey="value" stroke="none">
                            {chartData.map((entry, i) => <Cell key={i} fill={entry.color} />)}
                          </Pie>
                          <Tooltip contentStyle={{ backgroundColor: '#0F0F12', border: '1px solid #242429', fontFamily: 'IBM Plex Mono', fontSize: '11px' }} itemStyle={{ color: '#EDEDE7' }} />
                        </PieChart>
                      </ResponsiveContainer>
                      <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
                        <span className="font-['Saira_Stencil_One'] text-3xl text-[#EDEDE7]">{results.total_students}</span>
                        <span className="font-['IBM_Plex_Mono'] text-[8px] tracking-[1.5px] text-[#62626B] mt-1">TOTAL</span>
                      </div>
                    </div>
                    <div className="flex gap-4 mt-2">
                      <span className="font-['IBM_Plex_Mono'] text-[11px] text-[#9A9AA1] flex items-center gap-2"><i className="w-[9px] h-[9px] bg-[#FF4A3D]"></i>High Risk ({results.at_risk_count})</span>
                      <span className="font-['IBM_Plex_Mono'] text-[11px] text-[#9A9AA1] flex items-center gap-2"><i className="w-[9px] h-[9px] bg-[#3FD18B]"></i>On Track ({results.total_students - results.at_risk_count})</span>
                    </div>
                  </div>

                  <div className="bg-[#0F0F12] border border-[#242429] p-5 relative">
                    <span className="absolute top-0 right-0 w-2.5 h-2.5 border-t border-r border-[#46464F]"></span>
                    <h3 className="font-['Oswald'] text-[12px] tracking-[1.8px] text-[#9A9AA1] uppercase flex items-center gap-2 mb-4">
                      <svg className="w-[15px] h-[15px] stroke-[#62626B] fill-none" viewBox="0 0 24 24"><line x1="4" y1="20" x2="20" y2="20"/><polyline points="5 16 9 11 13 14 19 6"/></svg>
                      Batch Metrics
                    </h3>
                    <div className="space-y-4">
                      {[
                        { label: 'Total Records',    value: results.total_students,  color: '#EDEDE7' },
                        { label: 'Risk Density',     value: `${((results.at_risk_count / results.total_students) * 100).toFixed(1)}%`, color: '#FF4A3D' },
                        { label: 'Model CV AUC',     value: modelAccuracy ? `${modelAccuracy}%` : 'N/A', color: '#3FD18B' },
                        { label: 'Decision Threshold', value: results.threshold_used, color: '#FFB000' },
                        { label: 'Uploaded By',      value: results.uploaded_by,     color: '#9A9AA1' },
                      ].map(({ label, value, color }) => (
                        <div key={label} className="flex justify-between py-2 border-b border-[#242429]">
                          <span className="font-['IBM_Plex_Mono'] text-[11px] text-[#62626B]">{label}</span>
                          <span className="font-['IBM_Plex_Mono'] text-[13px]" style={{ color }}>{value}</span>
                        </div>
                      ))}
                    </div>
                    {modelAccuracy && (
                      <div className="mt-4 font-['IBM_Plex_Mono'] text-[11px] text-[#62626B] flex items-center gap-2">
                        <svg className="w-[13px] h-[13px] stroke-[#FFB000] fill-none stroke-[1.6] flex-none" viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><line x1="12" y1="11" x2="12" y2="16"/><circle cx="12" cy="8" r=".6"/></svg>
                        Model CV ROC-AUC: <span className="text-[#3FD18B]">{modelAccuracy}%</span>
                      </div>
                    )}
                  </div>

                  {/* Top risk drivers across whole batch */}
                  {results.data.some(s => s.risk_factors?.length) && (
                    <div className="md:col-span-2 bg-[#0F0F12] border border-[#242429] p-5">
                      <h3 className="font-['Oswald'] text-[12px] tracking-[1.8px] text-[#9A9AA1] uppercase mb-4">Top Risk Drivers Across Cohort</h3>
                      <div className="h-64">
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart
                            data={(() => {
                              const agg = {};
                              results.data.filter(s => s.is_at_risk).forEach(s =>
                                s.risk_factors?.forEach(f => { agg[f.feature] = (agg[f.feature] || 0) + Math.abs(f.impact); })
                              );
                              return Object.entries(agg)
                                .map(([feature, total]) => ({ feature, total: parseFloat(total.toFixed(2)) }))
                                .sort((a, b) => b.total - a.total).slice(0, 8);
                            })()}
                            layout="vertical" margin={{ top: 5, right: 30, left: 80, bottom: 5 }}
                          >
                            <CartesianGrid strokeDasharray="3 3" stroke="#242429" horizontal={false} />
                            <XAxis type="number" stroke="#62626B" tick={{ fill: '#62626B', fontSize: 10, fontFamily: 'IBM Plex Mono' }} />
                            <YAxis dataKey="feature" type="category" stroke="#62626B" tick={{ fill: '#9A9AA1', fontSize: 11, fontFamily: 'IBM Plex Mono' }} />
                            <Tooltip contentStyle={{ backgroundColor: '#0F0F12', border: '1px solid #242429', fontFamily: 'IBM Plex Mono', fontSize: '11px' }} />
                            <Bar dataKey="total" fill="#FFB000" barSize={14} />
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div className="bg-[#0F0F12] p-12 border border-dashed border-[#34343C] text-center">
                  <svg className="w-[46px] h-[46px] mx-auto mb-4 stroke-[#62626B] fill-none stroke-[1.3]" viewBox="0 0 24 24"><line x1="4" y1="20" x2="20" y2="20"/><polyline points="5 16 9 11 13 14 19 6"/></svg>
                  <h4 className="font-['Oswald'] font-medium text-[18px] tracking-[0.5px]">Awaiting Dataset</h4>
                  <p className="text-[#9A9AA1] text-[13px] mt-2">Return to Dashboard and initialize a prediction cycle.</p>
                </div>
              )}
            </div>
          )}

          {/* ── History Tab ── */}
          {activeTab === 'history' && (
            <div className="animate-in fade-in slide-in-from-bottom-4 duration-500 max-w-6xl">
              <div className="mb-6">
                <div className="font-['IBM_Plex_Mono'] text-[10px] tracking-[2.5px] text-[#FFB000] uppercase mb-2">Audit Trail</div>
                <h1 className="font-['Saira_Stencil_One'] text-4xl text-[#EDEDE7] tracking-wide mb-2">Batch History</h1>
                <p className="text-[#9A9AA1] text-[13px]">All previous prediction batches stored in the database.</p>
              </div>

              {historyLoading ? (
                <div className="font-['IBM_Plex_Mono'] text-[11px] text-[#62626B] animate-pulse tracking-widest">Loading records...</div>
              ) : historyByBatch.length === 0 ? (
                <div className="bg-[#0F0F12] p-12 border border-dashed border-[#34343C] text-center">
                  <h4 className="font-['Oswald'] text-[18px]">No history yet</h4>
                  <p className="text-[#9A9AA1] text-[13px] mt-2">Run a batch prediction to see records here.</p>
                </div>
              ) : (
                <div className="space-y-4">
                  {historyByBatch.map(batch => (
                    <div key={batch.batch_id} className="bg-[#0F0F12] border border-[#242429]">
                      <div className="p-4 border-b border-[#242429] flex items-center justify-between">
                        <div>
                          <span className="font-['IBM_Plex_Mono'] text-[12px] text-[#FFB000]">{batch.batch_id}</span>
                          <span className="font-['IBM_Plex_Mono'] text-[10px] text-[#62626B] ml-4">{batch.records.length} students · {batch.at_risk} at risk</span>
                        </div>
                        <span className={`font-['Oswald'] text-[10px] tracking-[1px] uppercase px-2 py-1 border ${batch.at_risk > 0 ? 'text-[#FF4A3D] border-[rgba(255,74,61,.4)] bg-[rgba(255,74,61,.08)]' : 'text-[#3FD18B] border-[rgba(63,209,139,.4)] bg-[rgba(63,209,139,.08)]'}`}>
                          {batch.at_risk > 0 ? `${batch.at_risk} flagged` : 'All clear'}
                        </span>
                      </div>
                      <div className="overflow-x-auto">
                        <table className="w-full text-left">
                          <thead>
                            <tr>
                              {['Student ID', 'Risk %', 'Status', 'Action', 'Date', 'Intervention'].map(h => (
                                <th key={h} className="font-['Oswald'] text-[10px] tracking-[1.5px] text-[#62626B] uppercase px-4 py-2 border-b border-[#242429]">{h}</th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {batch.records.map(r => (
                              <tr key={r.student_id} className="hover:bg-[#15151A] transition-colors">
                                <td className="px-4 py-2.5 border-b border-[#242429] font-['IBM_Plex_Mono'] text-[11px] text-[#62626B]">{r.student_id}</td>
                                <td className="px-4 py-2.5 border-b border-[#242429] font-['IBM_Plex_Mono'] text-[12px]" style={{ color: r.is_at_risk ? '#FF4A3D' : '#3FD18B' }}>{r.risk_probability}%</td>
                                <td className="px-4 py-2.5 border-b border-[#242429]">
                                  <span className={`font-['Oswald'] text-[10px] uppercase px-1.5 py-0.5 ${r.is_at_risk ? 'text-[#FF4A3D]' : 'text-[#3FD18B]'}`}>
                                    {r.is_at_risk ? 'At Risk' : 'Stable'}
                                  </span>
                                </td>
                                <td className="px-4 py-2.5 border-b border-[#242429]">
                                  {r.is_at_risk && (
                                    <button
                                      onClick={() => handleAction(r.student_id, r.actioned)}
                                      className={`font-['Oswald'] text-[9px] tracking-[1px] uppercase px-2 py-1 border transition-colors ${r.actioned ? 'text-[#3FD18B] border-[rgba(63,209,139,.4)]' : 'text-[#FFB000] border-[rgba(255,176,0,.4)] hover:bg-[rgba(255,176,0,.08)]'}`}
                                    >
                                      {r.actioned ? '✓ Actioned' : 'Mark Done'}
                                    </button>
                                  )}
                                </td>
                                <td className="px-4 py-2.5 border-b border-[#242429] font-['IBM_Plex_Mono'] text-[10px] text-[#62626B]">
                                  {r.created_at ? new Date(r.created_at).toLocaleDateString() : '—'}
                                </td>
                                <td className="px-4 py-2.5 border-b border-[#242429] font-['IBM_Plex_Sans'] text-[11px] text-[#9A9AA1] max-w-xs truncate">
                                  {r.intervention_plan || '—'}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

        </div>
      </main>
    </div>
  );
}

export default App;