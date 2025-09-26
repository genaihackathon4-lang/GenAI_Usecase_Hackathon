
// import './App.css';
// import StartupDocumentAnalyzer from './Components/startupdocumentanalyzer';


// function App() {
//   return (
//     <div>
//       <StartupDocumentAnalyzer/>
//     </div>
//   );
// }

// export default App;

// File: App.jsx
import React, { useState } from "react";
import StartupDocumentAnalyzer from './Components/startupdocumentanalyzer';
import StartupQuestionnaire from "./Components/StartupQuestionnaire";
import FileUpload from './Components/startupdocumentanalyzer';


function App() {
  const [userEmail, setUserEmail] = useState(null);
  const [uploadedData, setUploadedData] = useState(null);

  // Callback when upload is complete
  const handleUploadComplete = (email, data) => {
    setUserEmail(email);
    setUploadedData(data);
  };

  return (
    <div style={{ maxWidth: 900, margin: "0 auto", padding: 20 }}>
      {!userEmail ? (
        <StartupDocumentAnalyzer onUploadComplete={handleUploadComplete} />
      ) : (
        <StartupQuestionnaire userEmail={userEmail} uploadedData={uploadedData} />
      )}
    </div>
  );
}

export default App;

