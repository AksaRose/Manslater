import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import { Analytics } from "@vercel/analytics/react";
import Home from "./Home";
import Translator from "./Translator";
import "./App.css";

function App() {
  return (
    <>
      <Router>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/translator" element={<Translator />} />
        </Routes>
      </Router>
      <Analytics />
    </>
  );
}

export default App;
