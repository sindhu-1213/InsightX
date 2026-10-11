import React from 'react';
import { Outlet } from 'react-router-dom';
import Navbar from './Navbar';
import Footer from './Footer';

function Layout() {
  return (
    <div
      id="app-root-layout"
      style={{
        display: 'flex',
        flexDirection: 'column',
        minHeight: '100vh',
        backgroundColor: 'var(--bg-primary)',
        color: 'var(--text-primary)',
        position: 'relative',
        overflowX: 'hidden'
      }}
    >
      <Navbar />
      <main
        id="main-content-region"
        style={{
          flex: '1 0 auto',
          display: 'flex',
          flexDirection: 'column'
        }}
      >
        <Outlet />
      </main>
      <Footer />
    </div>
  );
}

export default Layout;
