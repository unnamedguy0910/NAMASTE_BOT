let mediaRecorder;
let audioChunks = [];

document.getElementById("startBtn").onclick = async () => {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  mediaRecorder = new MediaRecorder(stream);
  mediaRecorder.start();

  audioChunks = [];
  mediaRecorder.ondataavailable = event => {
    if (event.data.size > 0) {
      audioChunks.push(event.data);
    }
  };

  mediaRecorder.onstop = async () => {
    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
    const formData = new FormData();
    formData.append("file", audioBlob, "input.wav");

    const res = await fetch("http://127.0.0.1:8000/voice", {
      method: "POST",
      body: formData
    });

    const audioData = await res.blob();
    document.getElementById("replyAudio").src = URL.createObjectURL(audioData);
  };
};

document.getElementById("stopBtn").onclick = () => {
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
  }
};
