import React, { useCallback, useState } from "react";
import {
  Box,
  Button,
  Container,
  Typography,
  Paper,
  Grid,
  List,
  ListItem,
  ListItemText,
  LinearProgress,
  Card,
  CardContent,
  IconButton,
  TextField,
  Stack
} from "@mui/material";
import { Upload as UploadIcon, Delete as DeleteIcon, Print } from "@mui/icons-material";
import { useDropzone } from "react-dropzone";
import axios from "axios";
 
function humanFileSize(bytes) {
  if (bytes === 0) return "0 B";
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  const sizes = ["B", "KB", "MB", "GB", "TB"];
  return (bytes / Math.pow(1024, i)).toFixed(2) + " " + sizes[i];
}
 
export default function App() {
  const [files, setFiles] = useState([]); // File objects
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [report, setReport] = useState(null);
  const [error, setError] = useState(null);
  const [founder_email, setfounder_email] = useState("");
 
  // react-dropzone hook
  const onDrop = useCallback((acceptedFiles) => {
    // Merge new files; filter duplicates by name+size
    setFiles(prev => {
      const map = new Map(prev.map(f => [f.name + f.size, f]));
      acceptedFiles.forEach(f => {
        map.set(f.name + f.size, f);
      });
      return Array.from(map.values());
    });
    setReport(null);
  }, []);
 
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      // accept a wide range; adjust as needed
      "application/pdf": [".pdf"],
      "application/vnd.ms-powerpoint": [".ppt", ".pptx"],
      "application/vnd.openxmlformats-officedocument.presentationml.presentation": [".pptx"],
      "application/msword": [".doc", ".docx"],
      "text/plain": [".txt"],
      "audio/*": [],
      "video/*": []
    },
    multiple: true
  });
 
  // Directory picker (non-standard attribute). We'll use a hidden input to support folder selection.
  const handleSelectDirectory = (e) => {
    const fileList = Array.from(e.target.files || []);
    if (fileList.length) onDrop(fileList);
    // reset value so same folder can be picked again
    e.target.value = null;
    setReport(null);
  };
 
  const handleRemove = (idx) => {
    setFiles(prev => prev.filter((_, i) => i !== idx));
    setFiles(prev => {
      const updated = prev.filter((_, i) => i !== idx);
      if (updated.length === 0) {
        setReport(null);  // clear report if no files left
      }
      return updated;
   });
  };
 
  const handleClearAll = () => {
    setFiles([]);
    setReport(null);
    setError(null);
  };
 
  const handleUpload = async () => {
    if (!files.length) {
      setError("No files selected.");
      return;
    }
    if (!founder_email) {
      setError("Please enter your email before uploading.");
      return;
    }
    setUploading(true);
    setError(null);
    setReport(null);
    setProgress(0);
 
    try {
      const formData = new FormData();
   
      // append files
      files.forEach((f) => formData.append("files", f));
   
      // append user_email (FastAPI expects it as Form field)
      formData.append("founder_email", founder_email); // make sure userEmail comes from your state or input
   
      const resp = await axios.post(
        "https://hackathon-be-753168549263.us-central1.run.app/full-analysis",
        formData,
        {
          headers: { "Content-Type": "multipart/form-data" },
          onUploadProgress: (p) => {
            const percent = Math.round((p.loaded * 100) / (p.total || 1));
            setProgress(percent);
          },
          timeout: 5 * 60 * 1000, // extend if files are large
        }
      );
   
      setReport(resp.data);
      console.log(resp.data)
    } catch (err) {
      console.error(err);
      setError(err?.response?.data?.detail || err.message || "Upload failed");
    } finally {
      setUploading(false);
    }
  }
 
  return (
    <Container maxWidth="md" sx={{ py: 4 }}>
      <Typography variant="h4" gutterBottom align="center">
        AI Startup Analysis — Upload Files & Folders
      </Typography>
 
      <Paper elevation={3} sx={{ p: 3, mb: 3 }}>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} md={8}>
          <TextField
              label="Your Email"
              variant="outlined"
              fullWidth
              value={founder_email}
              onChange={(e) => {
                const email = e.target.value;
                setfounder_email(email);
 
                // Real-time Gmail validation
                const gmailRegex = /^[a-zA-Z0-9._%+-]+@gmail\.com$/;
                if (email && !gmailRegex.test(email)) {
                  setError("Please enter a valid Gmail address.");
                } else {
                  setError(null);
                }}
              }
              error={!!error} // highlights the input box if there's an error
              helperText={error}
              sx={{ mb: 2 }}
            />
            <div
              {...getRootProps()}
              style={{
                border: "2px dashed #1976d2",
                borderRadius: 8,
                padding: 20,
                textAlign: "center",
                cursor: "pointer",
                background: isDragActive ? "#e3f2fd" : "transparent"
              }}
            >
              <input {...getInputProps()} />
              <Typography variant="body1">
                {isDragActive ? "Drop the files here..." : "Drag & drop files here, or click to select files"}
              </Typography>
              <Typography variant="caption" display="block" sx={{ mt: 1 }}>
                Accepts: PDF, PPT, DOCX, video, audio. To upload a folder, use the "Select folder" button.
              </Typography>
            </div>
          </Grid>
 
          <Grid item xs={12} md={4}>
            <Stack spacing={1}>
              <Button
                variant="contained"
                component="label"
                startIcon={<UploadIcon />}
                fullWidth
              >
                Select folder
                {/* non-standard attribute webkitdirectory to pick a folder */}
                <input
                  type="file"
                  webkitdirectory="true"
                  directory="true"
                  multiple
                  hidden
                  onChange={handleSelectDirectory}
                />
              </Button>
 
              <Button variant="outlined" color="secondary" onClick={handleClearAll} fullWidth>
                Clear
              </Button>
            </Stack>
          </Grid>
        </Grid>
      </Paper>
 
      <Grid container spacing={2}>
        <Grid item xs={12} md={6}>
          <Card variant="outlined">
            <CardContent>
              <Typography variant="h6">Selected Files ({files.length})</Typography>
              <List dense>
                {files.map((f, i) => (
                  <ListItem
                    key={f.name + f.size + i}
                    secondaryAction={
                      <IconButton edge="end" aria-label="delete" onClick={() => handleRemove(i)}>
                        <DeleteIcon />
                      </IconButton>
                    }
                  >
                    <ListItemText
                      primary={f.name}
                      secondary={`${humanFileSize(f.size)} • ${f.type || "unknown"}`}
                    />
                  </ListItem>
                ))}
 
                {files.length === 0 && <Typography variant="body2" sx={{ mt: 1 }}>No files selected.</Typography>}
              </List>
 
              <Box sx={{ mt: 2 }}>
                <Button
                  variant="contained"
                  color="primary"
                  onClick={handleUpload}
                  disabled={uploading || files.length === 0 || !!report}
                  startIcon={<UploadIcon />}
                >
                  {uploading ? "Analyzing..." : "Send to Analysis"}
                </Button>
              </Box>
 
              {uploading && (
                <Box sx={{ mt: 2 }}>
                  <Typography variant="body2">Analysis in progress: {progress}%</Typography>
                  <LinearProgress variant="determinate" value={progress} sx={{ mt: 1 }} />
                </Box>
              )}
 
              {/* {error && <Typography color="error" sx={{ mt: 2 }}>{error}</Typography>} */}
              {error && (
                <Typography color="error" sx={{ mt: 2 }}>
                  {typeof error === "string"
                    ? error
                    : JSON.stringify(error, null, 2)}
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>
 
        <Grid item xs={12} md={6}>
          <Card variant="outlined">
            <CardContent>
              <Typography variant="h6">Analysis Report</Typography>
 
              {!report && <Typography variant="body2" sx={{ mt: 1 }}>No report yet. After upload, the structured report will appear here.</Typography>}
 
             {report?.response?.response && (
              <Box sx={{ mt: 2 }}>
                <Typography variant="h6" gutterBottom>
                  Startup Analysis Report
                </Typography>

                {Object.entries(report.response.response).map(([key, value]) => (
                  <Box key={key} sx={{ mb: 1 }}>
                    <Typography variant="subtitle2" sx={{ fontWeight: "bold", color: "primary.main" }}>
                      {key}
                    </Typography>
                    <Typography variant="body2">{value || "—"}</Typography>
                  </Box>
                ))}
              </Box>
            )}


            </CardContent>
          </Card>
        </Grid>
      </Grid>
 
      <Box sx={{ mt: 4, textAlign: "center" }}>
        <Typography variant="caption" color="text.secondary">
          Tip: For better UX with very large uploads, implement direct uploads to Cloud Storage with signed URLs and notify backend to process files from the bucket.
        </Typography>
      </Box>
    </Container>
  );
}