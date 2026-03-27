import { Routes, Route, NavLink } from 'react-router-dom';
import MainPage from './pages/MainPage';
import InterpretabilityPage from './pages/InterpretabilityPage';

export default function App() {
  return (
    <div className="app-container">
      <header className="app-header">
        <h1>
          <span className="icon">❄️</span>
          HX Frost Monitor
        </h1>
        <nav>
          <NavLink to="/" end className={({ isActive }) => isActive ? 'active' : ''}>
            Monitoring
          </NavLink>
          <NavLink to="/interpretability" className={({ isActive }) => isActive ? 'active' : ''}>
            Interpretability
          </NavLink>
        </nav>
      </header>
      <main className="app-main">
        <Routes>
          <Route path="/" element={<MainPage />} />
          <Route path="/interpretability" element={<InterpretabilityPage />} />
        </Routes>
      </main>
    </div>
  );
}
