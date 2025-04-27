import React, { useRef, useState } from 'react';
import { Canvas } from '@react-three/fiber';
import { ARButton, XR } from '@react-three/xr';
import { Html } from '@react-three/drei';
import axios from 'axios';

// Placeholder for camera capture: use a static image for now
const STATIC_IMAGE_URL = '/sample.jpg'; // Place a sample.jpg in public for demo

function VQAOverlay({ answer }) {
  return answer ? (
    <Html position={[0, 1.5, -2]} style={{ background: 'rgba(0,0,0,0.7)', color: 'white', padding: '1em', borderRadius: '8px' }}>
      <div>Answer: {answer}</div>
    </Html>
  ) : null;
}

export default function App() {
  const [answer, setAnswer] = useState(null);
  const [loading, setLoading] = useState(false);

  // Simulate camera capture by using a static image
  const handleVQATrigger = async () => {
    setLoading(true);
    setAnswer(null);
    try {
      // Fetch the static image as a blob
      const imageResponse = await fetch(STATIC_IMAGE_URL);
      const imageBlob = await imageResponse.blob();
      const formData = new FormData();
      formData.append('question', 'What is in the image?');
      formData.append('image_file', imageBlob, 'sample.jpg');
      const res = await axios.post('http://localhost:8000/vqa/', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setAnswer(res.data.answer);
    } catch (err) {
      setAnswer('Error: Could not get answer.');
    }
    setLoading(false);
  };

  return (
    <div style={{ width: '100vw', height: '100vh', overflow: 'hidden' }}>
      <ARButton />
      <Canvas style={{ width: '100vw', height: '100vh' }}>
        <XR>
          {/* AR Scene contents go here */}
          <ambientLight />
          <VQAOverlay answer={answer} />
        </XR>
      </Canvas>
      <button
        style={{ position: 'absolute', bottom: 40, left: '50%', transform: 'translateX(-50%)', zIndex: 10, padding: '1em 2em', fontSize: '1.2em', borderRadius: '8px', background: '#222', color: 'white' }}
        onClick={handleVQATrigger}
        disabled={loading}
      >
        {loading ? 'Asking...' : 'Ask VQA'}
      </button>
    </div>
  );
}
