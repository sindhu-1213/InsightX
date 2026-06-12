import React from 'react';

function Navbar() {
  return (
    <nav
      style={{
        height: '80px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '0 40px',
        backgroundColor: '#041428',
        borderBottom: '1px solid #0d2d4a'
      }}
    >
      <h2
        style={{
          color: '#00d4ff',
          margin: 0,
          fontFamily: 'Orbitron, sans-serif'
        }}
      >
        InsightX
      </h2>

      <div
        style={{
          display: 'flex',
          gap: '30px',
          color: '#ffffff'
        }}
      >
        <span>Home</span>
        <span>Features</span>
        <span>About</span>
        <span>Login</span>
      </div>
    </nav>
  );
}

export default Navbar;