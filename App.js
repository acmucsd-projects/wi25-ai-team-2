import React, { useEffect, useState } from 'react';
import {
  Platform,
  View,
  Text,
  Button,
  TextInput,
  StyleSheet,
  ActivityIndicator,
  ScrollView,
} from 'react-native';
import * as DocumentPicker from 'expo-document-picker';
import { JSONBIN_BIN_ID, JSONBIN_API_KEY } from '@env';

export default function App() {
  const [backendUrl, setBackendUrl] = useState('');
  const [fileStatus, setFileStatus] = useState('');
  const [query, setQuery] = useState('');
  const [answer, setAnswer] = useState('');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    // Fetch backend URL from JSONBin once on mount
    const fetchBackendUrl = async () => {
      try {
        const res = await fetch(`https://api.jsonbin.io/v3/b/${JSONBIN_BIN_ID}/latest`, {
          headers: {
            'X-Master-Key': JSONBIN_API_KEY,
            'Content-Type': 'application/json',
          },
        });

        if (!res.ok) throw new Error(`HTTP error ${res.status}`);

        const json = await res.json();
        const url = json.record?.url;

        if (!url) throw new Error('Backend URL not found in JSONBin record');

        setBackendUrl(url);
      } catch (error) {
        console.error('Failed to load backend URL:', error);
        setFileStatus('Error loading backend URL');
      }
    };

    fetchBackendUrl();
  }, []);

  const pickAndUploadFileMobile = async () => {
    if (!backendUrl) {
      setFileStatus('Backend URL not loaded yet');
      return;
    }
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

      const response = await fetch(`${backendUrl}/upload/`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Upload failed: HTTP ${response.status} - ${errorText}`);
      }

      const data = await response.json();
      setFileStatus(data.message || 'Upload complete');
    } catch (err) {
      console.error(err);
      setFileStatus('Upload failed: ' + (err.message || err));
    }
  };

  const pickAndUploadFileWeb = async (event) => {
    if (!backendUrl) {
      setFileStatus('Backend URL not loaded yet');
      return;
    }

    const file = event.target.files[0];
    if (!file) return;

    setFileStatus('Uploading...');

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${backendUrl}/upload/`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Upload failed: HTTP ${response.status} - ${errorText}`);
      }

      const data = await response.json();
      setFileStatus(data.message || 'Upload complete');
    } catch (err) {
      console.error(err);
      setFileStatus('Upload failed: ' + (err.message || err));
    }
  };

  const submitQuery = async () => {
    if (!query.trim()) return;
    if (!backendUrl) {
      setAnswer('Backend URL not loaded yet');
      return;
    }

    setLoading(true);
    setAnswer('');

    try {
      const response = await fetch(`${backendUrl}/query/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Query failed: HTTP ${response.status} - ${errorText}`);
      }

      const data = await response.json();
      setAnswer(data.answer || 'No answer returned');
    } catch (err) {
      console.error(err);
      setAnswer('Error retrieving answer: ' + (err.message || err));
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
            disabled={!backendUrl}
          />
          <Text>{fileStatus}</Text>
        </>
      ) : (
        <>
          <Button
            title="Upload File"
            onPress={pickAndUploadFileMobile}
            disabled={!backendUrl}
          />
          <Text>{fileStatus}</Text>
        </>
      )}

      <TextInput
        style={styles.input}
        placeholder="Enter your question..."
        value={query}
        onChangeText={setQuery}
        editable={!!backendUrl}
      />
      <Button
        title="Submit Query"
        onPress={submitQuery}
        disabled={!query.trim() || loading || !backendUrl}
      />

      {loading ? (
        <ActivityIndicator style={{ marginTop: 20 }} size="large" />
      ) : (
        <Text style={styles.answer}>{answer}</Text>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flexGrow: 1, padding: 20, justifyContent: 'center' },
  title: { fontSize: 24, fontWeight: 'bold', marginBottom: 20, textAlign: 'center' },
  input: { borderWidth: 1, borderColor: '#999', padding: 10, marginTop: 20, marginBottom: 10, borderRadius: 5 },
  answer: { marginTop: 20, fontSize: 16 },
});
