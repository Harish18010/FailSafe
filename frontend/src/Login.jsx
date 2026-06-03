import { useState } from 'react';
import axios from 'axios';

export default function Login({ setToken }) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    
    try {
      const formData = new URLSearchParams();
      formData.append('username', username);
      formData.append('password', password);
      
      const response = await axios.post('http://127.0.0.1:8000/token', formData, {
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' }
      });
      
      const token = response.data.access_token;
      localStorage.setItem('token', token);
      setToken(token);
    } catch (err) {
      setError('Invalid username or password.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#0f1011] flex items-center justify-center p-4 font-sans text-gray-200">
      <div className="w-full max-w-md bg-[#1a1b1e] border border-gray-800 rounded-2xl p-8 shadow-2xl relative overflow-hidden">
        <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-orange-500 to-red-600"></div>
        
        <div className="flex flex-col items-center mb-8">
          <div className="w-12 h-12 bg-gradient-to-br from-orange-500 to-red-600 rounded-xl flex items-center justify-center font-bold text-white text-xl shadow-lg shadow-orange-500/20 mb-4">
            F
          </div>
          <h2 className="text-2xl font-bold text-white tracking-tight">FAILSAFE</h2>
          <p className="text-xs text-gray-500 font-medium uppercase tracking-widest mt-1">Authentication Required</p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-5">
          <div>
            <label className="block text-xs font-bold text-gray-500 uppercase tracking-wider mb-2">Username</label>
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              className="w-full bg-[#0f1011] border border-gray-800 text-white px-4 py-3 rounded-lg focus:outline-none focus:border-orange-500 transition shadow-inner"
              required
            />
          </div>

          <div>
            <label className="block text-xs font-bold text-gray-500 uppercase tracking-wider mb-2">Password</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full bg-[#0f1011] border border-gray-800 text-white px-4 py-3 rounded-lg focus:outline-none focus:border-orange-500 transition shadow-inner"
              required
            />
          </div>

          {error && (
            <div className="p-3 bg-red-500/10 border border-red-500/20 rounded-lg text-xs font-bold text-red-500 text-center">
              {error}
            </div>
          )}

          <button
            type="submit"
            disabled={loading}
            className="w-full py-3.5 bg-white hover:bg-gray-200 text-black font-bold rounded-lg text-xs uppercase tracking-widest transition-all active:scale-95 disabled:opacity-50 mt-4 shadow-[0_0_15px_rgba(255,255,255,0.1)]"
          >
            {loading ? "Authenticating..." : "Establish Secure Session"}
          </button>
        </form>
      </div>
    </div>
  );
}