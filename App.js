import React, { useState, useRef } from "react";
import {
  View,
  Text,
  Button,
  TextInput,
  StyleSheet,
  ActivityIndicator,
  ScrollView,
  Alert,
  Platform,
} from "react-native";
import * as DocumentPicker from "expo-document-picker";
import { LOCAL_BACKEND_URL } from "@env";

export default function App() {
  const backendUrl = LOCAL_BACKEND_URL;
  const fileInputRef = useRef(null);

  const [fileStatus, setFileStatus] = useState("");
  const [query, setQuery] = useState("");
  const [answer, setAnswer] = useState("");
  const [loading, setLoading] = useState(false);

  const pickAndUploadFile = async () => {
    if (!backendUrl) {
      Alert.alert("Backend URL not loaded yet");
      return;
    }

    try {
      const res = await DocumentPicker.getDocumentAsync({
        copyToCacheDirectory: true,
      });
      if (res.type === "cancel") return;

      setFileStatus("Uploading...");
      const uri = res.uri || res.assets?.[0]?.uri;
      const fileName = res.name || res.assets?.[0]?.name;
      const fileType =
        res.mimeType || res.assets?.[0]?.mimeType || "application/octet-stream";

      const formData = new FormData();
      formData.append("file", {
        uri,
        name: fileName,
        type: fileType,
      });

      const response = await fetch(`${backendUrl}/upload/`, {
        method: "POST",
        headers: {
          "Content-Type": "multipart/form-data",
        },
        body: formData,
      });

      const data = await response.json();
      setFileStatus(data.message || "Upload successful");
    } catch (err) {
      console.error(err);
      setFileStatus("Upload failed.");
    }
  };

  const handleWebFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setFileStatus("Uploading...");

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${backendUrl}/upload/`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      setFileStatus(data.message || "Upload successful");
    } catch (err) {
      console.error(err);
      setFileStatus("Upload failed.");
    }
  };

  const triggerWebUpload = () => {
    fileInputRef.current?.click();
  };

  const submitQuery = async () => {
    if (!query.trim()) return;

    if (!backendUrl) {
      Alert.alert("Backend URL not loaded yet");
      return;
    }

    setLoading(true);
    setAnswer("");

    try {
      const response = await fetch(`${backendUrl}/query/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });

      const data = await response.json();
      setAnswer(data.answer || "No answer received");
    } catch (err) {
      console.error(err);
      setAnswer("Error retrieving answer.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.title}>Document QA App</Text>

      <Button
        title="Upload File"
        onPress={Platform.OS === "web" ? triggerWebUpload : pickAndUploadFile}
      />
      {Platform.OS === "web" && (
        <input
          type="file"
          ref={fileInputRef}
          style={{ display: "none" }}
          onChange={handleWebFileUpload}
        />
      )}
      <Text>{fileStatus}</Text>

      <TextInput
        style={styles.input}
        placeholder="Enter your question..."
        value={query}
        onChangeText={setQuery}
      />
      <Button title="Submit Query" onPress={submitQuery} />

      {loading ? (
        <ActivityIndicator style={{ marginTop: 20 }} />
      ) : (
        <Text style={styles.answer}>{answer}</Text>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flexGrow: 1, padding: 20, justifyContent: "center" },
  title: { fontSize: 24, fontWeight: "bold", marginBottom: 20 },
  input: {
    borderWidth: 1,
    padding: 10,
    marginTop: 20,
    marginBottom: 10,
    borderRadius: 6,
  },
  answer: { marginTop: 20, fontSize: 16 },
});
