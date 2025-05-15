import React, { useState } from 'react';
import { View, Text, Button, TextInput, StyleSheet, ActivityIndicator, ScrollView } from 'react-native';
import * as DocumentPicker from 'expo-document-picker';
import * as FileSystem from 'expo-file-system';

const API_BASE = 'https://1234-56-789-012-345.ngrok-free.app'; // get real link from google colab

export default function App() {
  const [fileStatus, setFileStatus] = useState('');
  const [query, setQuery] = useState('');
  const [answer, setAnswer] = useState('');
  const [loading, setLoading] = useState(false);

  const pickAndUploadFile = async () => {
    try {
      const res = await DocumentPicker.getDocumentAsync({ copyToCacheDirectory: true });
      if (res.type === 'cancel') return;

      setFileStatus('Uploading...');
      const uri = res.assets[0].uri;
      const fileName = res.assets[0].name;
      const fileType = res.assets[0].mimeType || 'application/octet-stream';

      const fileData = await FileSystem.readAsStringAsync(uri, { encoding: FileSystem.EncodingType.Base64 });

      const formData = new FormData();
      formData.append('file', {
        uri,
        name: fileName,
        type: fileType
      });

      const response = await fetch(`${API_BASE}/upload/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        body: formData
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
        body: JSON.stringify({ query })
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

      <Button title="Upload File" onPress={pickAndUploadFile} />
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
  container: { flexGrow: 1, padding: 20, justifyContent: 'center' },
  title: { fontSize: 24, fontWeight: 'bold', marginBottom: 20 },
  input: { borderWidth: 1, padding: 10, marginTop: 20, marginBottom: 10 },
  answer: { marginTop: 20, fontSize: 16 }
});
