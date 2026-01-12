import React, { Suspense, lazy } from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import { Analytics } from "@vercel/analytics/react";
import "./App.css";

// Lazy load components for code splitting - improves initial load time
const Home = lazy(() => import("./Home"));
const Translator = lazy(() => import("./Translator"));
const Convo = lazy(() => import("./Convo"));

// Loading fallback component
const LoadingFallback = () => (
  <div style={{
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    height: '100vh',
    backgroundColor: '#feabc1'
  }}>
    <div style={{ fontSize: '18px', color: '#880e4f' }}>Loading...</div>
  </div>
);

function App() {
  return (
    <>
      <Router>
        <Suspense fallback={<LoadingFallback />}>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/translator" element={<Translator />} />
            <Route path="/convo" element={<Convo />} />
          </Routes>
        </Suspense>
      </Router>
      <Analytics />
    </>
  );
}

export default App;
