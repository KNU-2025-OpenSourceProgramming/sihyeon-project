const { useState, useRef, useEffect } = React;
const {
  AppBar, Toolbar, Typography, Button, Container, Box, Paper, TextField
} = MaterialUI;

function App() {
  // 현재 접속한 URL을 기반으로 WebSocket URL 자동 생성
  const getDefaultWebsocketUrl = () => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    return `${protocol}//${host}/audio`;
  };

  const [isRecording, setIsRecording] = useState(false);
  const [transcriptions, setTranscriptions] = useState([]);
  const [websocketUrl, setWebsocketUrl] = useState(getDefaultWebsocketUrl());
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const socketRef = useRef(null);

  const handleWebSocketUrlChange = (event) => {
    setWebsocketUrl(event.target.value);
  };

  const handleStartRecording = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorderRef.current = new MediaRecorder(stream);

    mediaRecorderRef.current.ondataavailable = (event) => {
      audioChunksRef.current.push(event.data);
    };

    mediaRecorderRef.current.onstop = () => {
      sendAudioData();
    };

    mediaRecorderRef.current.start();
    setIsRecording(true);
  };

  const handleStopRecording = () => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const sendAudioData = () => {
    const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
    const reader = new FileReader();
    reader.onloadend = () => {
      const audioArrayBuffer = reader.result;
      if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
        socketRef.current.send(audioArrayBuffer);
      }
      audioChunksRef.current = [];
    };
    reader.readAsArrayBuffer(audioBlob);
  };

  const setupWebSocket = () => {
    socketRef.current = new WebSocket(websocketUrl);

    socketRef.current.onopen = () => {
      console.log('WebSocket is connected.');
    };

    socketRef.current.onmessage = (event) => {
      setTranscriptions((prev) => [...prev, event.data]);
    };

    socketRef.current.onclose = (event) => {
      console.log('WebSocket is closed.', event);
    };

    socketRef.current.onerror = (error) => {
      console.log('WebSocket error:', error);
    };
  };

  useEffect(() => {
    setupWebSocket();
    return () => {
      if (socketRef.current) {
        socketRef.current.close();
      }
    };
  }, [websocketUrl]);

  return React.createElement(
    Container,
    null,
    React.createElement(
      AppBar,
      { position: 'static' },
      React.createElement(
        Toolbar,
        null,
        React.createElement(Typography, { variant: 'h6' }, "Audio Recorder")
      )
    ),
    React.createElement(
      Box,
      { mt: 2 },
      React.createElement(TextField, {
        label: "WebSocket URL",
        variant: "outlined",
        fullWidth: true,
        value: websocketUrl,
        onChange: handleWebSocketUrlChange,
        style: { marginBottom: 16 }
      }),
      React.createElement(
        Button,
        {
          variant: 'contained',
          color: 'primary',
          onClick: handleStartRecording,
          disabled: isRecording
        },
        "Start Recording"
      ),
      React.createElement(
        Button,
        {
          variant: 'contained',
          color: 'secondary',
          onClick: handleStopRecording,
          disabled: !isRecording,
          style: { marginLeft: 16 }
        },
        "Stop Recording"
      )
    ),
    React.createElement(
      'div',
      { className: 'transcription-container' },
      transcriptions.map((text, index) =>
        React.createElement(
          'div',
          {
            key: index,
            className: 'chat-bubble'
          },
          text
        )
      )
    )
  );
}

ReactDOM.render(
  React.createElement(App),
  document.getElementById('root')
);