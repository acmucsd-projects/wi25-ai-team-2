import React, { useState } from 'react';
import { Platform, View, Text, Button, TextInput, StyleSheet, ActivityIndicator, ScrollView } from 'react-native';

// For mobile
import * as DocumentPicker from 'expo-document-picker';

// Dynamic backend URL from config.js
import { API_BASE } from './config'; // adjust if needed

export default function App() {
  const [fileStatus, setFileStatus] = useState('');
  const [query, setQuery] = useState('');
  const [answer, setAnswer] = useState('');
  const [loading, setLoading] = useState(false);

  // Upload for Mobile using expo-document-picker
  const pickAndUploadFileMobile = async () => {
    try {
      const res = await DocumentPicker.getDocumentAsync({ copyToCacheDirectory: true });
      if (res.type === 'cancel') return;

      setFileStatus('Uploading...');
      const { uri, name, mimeType } = res;

      const formData = new FormData();
      formData.append('file', {
        uri,
        name,
        type: mimeType || 'application/octet-stream',
      });

      // IMPORTANT: Don't set 'Content-Type', fetch will set it with correct boundary on React Native
      const response = await fetch(`${API_BASE}/upload/`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      setFileStatus(data.message);
    } catch (err) {
      console.error(err);
      setFileStatus('Upload failed.');
    }
  };

  // Upload for Web using input element
  const pickAndUploadFileWeb = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setFileStatus('Uploading...');

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${API_BASE}/upload/`, {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      setFileStatus(data.message);
    } catch (err) {
      console.error(err);
      setFileStatus('Upload failed.');
    }
  };

  const submitQuery = async () => {
    if (!query.trim()) return;

    setLoading(true);
    setAnswer('');

    try {
      const response = await fetch(`${API_BASE}/query/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
      });

      const data = await response.json();
      setAnswer(data.answer);
    } catch (err) {
      console.error(err);
      setAnswer('Error retrieving answer.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.title}>Document QA App</Text>

      {Platform.OS === 'web' ? (
        <>
          <input
            type="file"
            onChange={pickAndUploadFileWeb}
            style={{ marginBottom: 10 }}
          />
          <Text>{fileStatus}</Text>
        </>
      ) : (
        <>
          <Button title="Upload File" onPress={pickAndUploadFileMobile} />
          <Text>{fileStatus}</Text>
        </>
      )}

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
  container: { flexGrow: 1, padding: 20, justifyContent: 'center' },
  title: { fontSize: 24, fontWeight: 'bold', marginBottom: 20 },
  input: { borderWidth: 1, padding: 10, marginTop: 20, marginBottom: 10 },
  answer: { marginTop: 20, fontSize: 16 },
});
