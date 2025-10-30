import React, { useState, useEffect } from "react";

/**
 * AuthForm.jsx
 * Formular universal pentru login și înregistrare utilizatori.
 * 
 * Props:
 *   - type: "login" sau "register"
 *   - onAuth: funcție callback (primește obiect {username, token} sau null)
 */
function AuthForm({ type = "login", onAuth }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [loading, setLoading] = useState(false);

  // Clear inputs and messages when switching between login/register forms
  useEffect(() => {
    setUsername("");
    setPassword("");
    setError(null);
    setSuccess(null);
  }, [type]);

  // Automatically clear error messages after 4 seconds
  useEffect(() => {
    if (error) {
      const timer = setTimeout(() => setError(null), 4000);
      return () => clearTimeout(timer);
    }
  }, [error]);

  // Automatically clear success messages after 4 seconds
  useEffect(() => {
    if (success) {
      const timer = setTimeout(() => setSuccess(null), 4000);
      return () => clearTimeout(timer);
    }
  }, [success]);

  // Simple client-side validation for username and password
  const validate = () => {
    if (username.trim().length < 3) {
      setError("Username trebuie să aibă minim 3 caractere.");
      return false;
    }
    if (password.length < 5) {
      setError("Parola trebuie să aibă minim 5 caractere.");
      return false;
    }
    // Optional: Add regex validation for username/password complexity here
    return true;
  };

  // Handle form submission for both login and register
  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setSuccess(null);

    if (!validate()) return;

    setLoading(true);
    try {
      let url;
      let body;
      let headers;

      if (type === "login") {
        // For login, send x-www-form-urlencoded (OAuth2 password flow expected by FastAPI)
        url = "http://localhost:8000/users/login";
        body = new URLSearchParams();
        body.append("username", username.trim());
        body.append("password", password);
        headers = {};
      } else {
        // For registration, send JSON
        url = "http://localhost:8000/users/register";
        body = JSON.stringify({ username: username.trim(), password });
        headers = { "Content-Type": "application/json" };
      }

      const res = await fetch(url, {
        method: "POST",
        headers,
        body,
      });

      const data = await res.json();

      if (res.ok) {
        if (type === "login") {
          setSuccess("Autentificare reușită!");
          // Store username and token (can extend to localStorage for persistence)
          if (onAuth)
            onAuth({
              username,
              token: data.access_token,
            });
          // Optional localStorage usage:
          // localStorage.setItem("authToken", data.access_token);
          // localStorage.setItem("username", username);
        } else {
          setSuccess("Cont creat cu succes! Te poți autentifica acum.");
          setUsername("");
          setPassword("");
          if (onAuth) onAuth(null); // Reset auth state after registration
        }
      } else {
        // Detailed backend error handling
        if (typeof data === "object" && data) {
          if (data.detail) setError(data.detail);
          else if (Array.isArray(data.msg) && data.msg.length > 0)
            setError(data.msg[0]);
          else setError("Eroare la autentificare/înregistrare.");
        } else {
          setError("Eroare la autentificare/înregistrare.");
        }
      }
    } catch {
      setError("Eroare de rețea sau server, încearcă din nou.");
    } finally {
      setLoading(false);
    }
  };

  // Optional: Reset form button
  const handleReset = () => {
    setUsername("");
    setPassword("");
    setError(null);
    setSuccess(null);
  };

  return (
    <form
      onSubmit={handleSubmit}
      style={{
        margin: "32px auto",
        maxWidth: 350,
        background: "#f4f8fd",
        padding: 18,
        borderRadius: 10,
        boxShadow: "0 4px 14px #aad4ff21",
      }}
      autoComplete="on"
      aria-label={type === "login" ? "Formular autentificare" : "Formular înregistrare"}
    >
      <h2 style={{ textAlign: "center", color: "#2196f3" }}>
        {type === "login" ? "Autentificare" : "Înregistrare"}
      </h2>

      {/* Username input */}
      <label htmlFor="username" style={{ display: "none" }}>
        Username
      </label>
      <input
  id="username"
  type="text"
  autoComplete="username"
  placeholder="Utilizator"
  autoFocus
  required
  value={username}
  onChange={(e) => setUsername(e.target.value)}
  onKeyDown={(e) => e.key === "Enter" && handleSubmit(e)}
  disabled={loading}
  aria-invalid={!!error}
  style={{
    marginBottom: 10,
    width: "100%",
    padding: "8px 8px",        // padding uniform stânga-dreapta
    fontSize: 15,
    border: `1px solid ${error ? "#e53935" : "#2196f3"}`,
    borderRadius: 6,
    outline: "none",
    boxSizing: "border-box",   // ca și la restul inputurilor!
  }}
/>


      {/* Password input with show/hide toggle */}
      <label htmlFor="password" style={{ display: "none" }}>
        Parolă
      </label>
      <div style={{ position: "relative", marginBottom: 12 }}>
        <input
          id="password"
          type={showPassword ? "text" : "password"}
          autoComplete={type === "login" ? "current-password" : "new-password"}
          placeholder="Parolă"
          required
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSubmit(e)}
          disabled={loading}
          aria-invalid={!!error}
          style={{
                 width: "100%",
                 padding: "8px 54px 8px 8px",
                 fontSize: 15,
                 border: `1px solid ${error ? "#e53935" : "#2196f3"}`,
                 borderRadius: 6,
                 outline: "none",
                boxSizing: "border-box",
          }}
        />
        <button
          type="button"
          tabIndex={-1}
          title={showPassword ? "Ascunde parola" : "Afișează parola"}
          aria-label={showPassword ? "Ascunde parola" : "Afișează parola"}
          onClick={() => setShowPassword((prev) => !prev)}
         style={{
  position: "absolute",
  right: 12,
  top: "50%",
  transform: "translateY(-50%)",
  background: "none",
  border: "none",
  outline: "none",
  color: "#2196f3",
  fontSize: 19,
  cursor: "pointer",
  userSelect: "none",
  height: 24,
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  padding: 0,
}}


        >
          {showPassword ? "🙈" : "👁️"}
        </button>
      </div>

      {/* Error message */}
      {error && (
        <div
          role="alert"
          aria-live="assertive"
          style={{
            marginTop: 10,
            color: "#e53935",
            fontWeight: "bold",
            textAlign: "center",
          }}
        >
          {error}
        </div>
      )}

      {/* Success message */}
      {success && (
        <div
          role="status"
          aria-live="polite"
          style={{
            marginTop: 10,
            color: "#43a047",
            fontWeight: "bold",
            textAlign: "center",
          }}
        >
          {success}
        </div>
      )}

      {/* Submit and Reset buttons */}
      <div style={{ display: "flex", gap: "10px", marginTop: 12 }}>
        <button
          type="submit"
          disabled={loading}
          aria-busy={loading}
          style={{
            flex: 1,
            padding: "7px 0",
            fontSize: 15,
            backgroundColor: "#2196f3",
            color: "#fff",
            fontWeight: "bold",
            border: "none",
            borderRadius: 6,
            boxShadow: "0 2px 8px #2196f366",
            cursor: loading ? "not-allowed" : "pointer",
            userSelect: "none",
          }}
        >
          {loading ? (
            <>
              <span className="loader" style={{ marginRight: 9 }}>
                &#9696;
              </span>
              {type === "login" ? "Se autentifică..." : "Se creează..."}
            </>
          ) : type === "login" ? (
            "Autentificare"
          ) : (
            "Înregistrare"
          )}
        </button>
        <button
          type="button"
          onClick={handleReset}
          disabled={loading}
          style={{
            flex: 1,
            padding: "7px 0",
            fontSize: 15,
            backgroundColor: "#f44336",
            color: "#fff",
            fontWeight: "bold",
            border: "none",
            borderRadius: 6,
            boxShadow: "0 2px 8px #f4433633",
            cursor: loading ? "not-allowed" : "pointer",
            userSelect: "none",
          }}
        >
          Resetare
        </button>
      </div>

      {/* Placeholder for forgot password */}
      {type === "login" && (
        <div style={{ marginTop: 11, textAlign: "right" }}>
          <a
            href="#"
            onClick={(e) => {
              e.preventDefault();
              alert("Funcția de resetare a parolei va fi disponibilă în curând.");
            }}
            style={{
              color: "#2196f3",
              fontSize: 12,
              textDecoration: "underline",
              cursor: "pointer",
              userSelect: "none",
            }}
          >
            Ai uitat parola?
          </a>
        </div>
      )}
    </form>
  );
}

export default AuthForm;
