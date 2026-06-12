import React from "react";

function LoginPage() {
  return (
    <div
      style={{
        minHeight: "100vh",
        backgroundColor: "#020b18",
        color: "#ffffff",
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        padding: "20px",
      }}
    >
      <div
        style={{
          width: "420px",
          backgroundColor: "#0a1f35",
          padding: "25px",
          borderRadius: "12px",
        }}
      >
        <h2 style={{ textAlign: "center", marginBottom: "20px" }}>
          Medical Access Portal
        </h2>

        {/* Email */}
        <input
          placeholder="Email"
          style={inputStyle}
        />

        {/* Password */}
        <input
          type="password"
          placeholder="Password"
          style={inputStyle}
        />

        {/* Role */}
        <select style={inputStyle}>
          <option>Doctor</option>
          <option>Hospital Staff</option>
        </select>

        {/* Medical ID */}
        <input
          placeholder="Medical License / Registration ID"
          style={inputStyle}
        />

        {/* Hospital Name */}
        <input
          placeholder="Hospital / Institution Name"
          style={inputStyle}
        />

        {/* Upload Doctor ID */}
        <label style={labelStyle}>Upload Doctor ID Proof</label>
        <input type="file" style={inputStyle} />

        {/* Upload Hospital ID */}
        <label style={labelStyle}>Upload Hospital ID Proof</label>
        <input type="file" style={inputStyle} />

        {/* Login Button */}
        <button style={buttonStyle}>
          Verify & Login
        </button>
      </div>
    </div>
  );
}

const inputStyle = {
  width: "100%",
  padding: "10px",
  margin: "8px 0",
  borderRadius: "6px",
  border: "none",
};

const labelStyle = {
  fontSize: "12px",
  marginTop: "10px",
  display: "block",
  color: "#8ab4c9",
};

const buttonStyle = {
  width: "100%",
  padding: "10px",
  marginTop: "15px",
  backgroundColor: "#00d4ff",
  border: "none",
  borderRadius: "6px",
  cursor: "pointer",
  fontWeight: "bold",
};

export default LoginPage;