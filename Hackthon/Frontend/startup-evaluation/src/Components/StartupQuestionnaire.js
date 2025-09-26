// File: StartupQuestionnaire.jsx
import React, { useState, useEffect, useRef } from "react";
import io from "socket.io-client";

const SOCKET_SERVER_URL = "https://8000-genaihackat-genaiusecas-wqojhy5zxzd.ws-us121.gitpod.io"; // Replace with your backend URL

const StartupQuestionnaire = ({ userEmail }) => {
  const [socket, setSocket] = useState(null);
  const [currentQuestion, setCurrentQuestion] = useState(null);
  const [answers, setAnswers] = useState({});
  const [finalJson, setFinalJson] = useState(null);
  const [listening, setListening] = useState(false);

  const recognitionRef = useRef(null);

  useEffect(() => {
    // Connect to Socket.IO backend
    const newSocket = io(SOCKET_SERVER_URL, { 
  path: "/ws",        // 👈 must match backend mount
  transports: ["websocket"] 
});

    setSocket(newSocket);

    newSocket.on("connect", () => {
      console.log("Connected to backend:", newSocket.id);
    });

    // Receive new question
    newSocket.on("new_question", (data) => {
      setCurrentQuestion(data);
      speakQuestion(data.text);
    });

    // Receive final JSON
    newSocket.on("final_json", (filledJson) => {
      setFinalJson(filledJson);
      setCurrentQuestion(null);
    });

    return () => {
      newSocket.disconnect();
    };
  }, []);

  // Function to speak the question using browser TTS
  const speakQuestion = (text) => {
    const utterance = new SpeechSynthesisUtterance(text);
    window.speechSynthesis.speak(utterance);
  };

  // Start voice recognition
  const startListening = () => {
    if (!("webkitSpeechRecognition" in window || "SpeechRecognition" in window)) {
      alert("Speech Recognition API not supported in this browser.");
      return;
    }
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const recognition = new SpeechRecognition();
    recognitionRef.current = recognition;
    recognition.lang = "en-US";
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    recognition.onstart = () => setListening(true);
    recognition.onend = () => setListening(false);

    recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript;
      submitAnswer(transcript);
    };

    recognition.start();
  };

  // Submit answer (either voice or text)
//   const submitAnswer = (answer) => {
//     if (!socket || !currentQuestion) return;
//     socket.emit("answer", { answer, user_email: userEmail });
//     setAnswers((prev) => ({ ...prev, [currentQuestion]: answer }));
//     setCurrentQuestion(null);
//   };
const submitAnswer = (answer) => {
  if (!socket || !currentQuestion) return;
  socket.emit("answer", { 
    answer, 
    user_email: userEmail, 
    key: currentQuestion.key   // send key back
  });
  setAnswers((prev) => ({ ...prev, [currentQuestion.text]: answer }));
  setCurrentQuestion(null);
};


  return (
    <div style={{ maxWidth: 600, margin: "0 auto", fontFamily: "Arial, sans-serif" }}>
      <h2>Startup Questionnaire</h2>

      {!finalJson && currentQuestion && (
        <div style={{ marginTop: 20 }}>
          <p><strong>Question:</strong> {currentQuestion.text}</p>
          <button onClick={startListening} disabled={listening}>
            {listening ? "Listening..." : "Answer by Voice"}
          </button>
          <div style={{ marginTop: 10 }}>
            <input
              type="text"
              placeholder="Answer by typing..."
              onKeyDown={(e) => {
                if (e.key === "Enter") submitAnswer(e.target.value);
              }}
              style={{ width: "100%", padding: 8 }}
            />
          </div>
        </div>
      )}

      {!finalJson && !currentQuestion && <p>Waiting for next question...</p>}

      {finalJson && (
        <div style={{ marginTop: 20 }}>
          <h3>All Questions Answered!</h3>
          <pre style={{ background: "#f4f4f4", padding: 10 }}>{JSON.stringify(finalJson, null, 2)}</pre>
        </div>
      )}

      <div style={{ marginTop: 20 }}>
        <h4>Answers so far:</h4>
        <ul>
          {Object.entries(answers).map(([q, a]) => (
            <li key={q}><strong>{q}:</strong> {a}</li>
          ))}
        </ul>
      </div>
    </div>
  );
};

export default StartupQuestionnaire;
