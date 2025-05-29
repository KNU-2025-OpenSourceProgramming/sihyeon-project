const { useState, useEffect } = React;

function App() {
    const [searchQuery, setSearchQuery] = useState('');
    const [videos, setVideos] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    const searchVideos = async () => {
        if (!searchQuery.trim()) return;
        
        setLoading(true);
        setError('');
        
        try {
            const response = await fetch(`/api/search/${encodeURIComponent(searchQuery)}`);
            const data = await response.json();
            
            if (response.ok) {
                setVideos(data.results);
            } else {
                setError(data.error || 'An error occurred');
            }
        } catch (err) {
            setError('Failed to fetch videos');
        } finally {
            setLoading(false);
        }
    };

    const handleKeyPress = (e) => {
        if (e.key === 'Enter') {
            searchVideos();
        }
    };

    const openVideo = (videoId) => {
        window.open(`https://www.youtube.com/watch?v=${videoId}`, '_blank');
    };

    return (
        <div className="container">
            <h1>YouTube Video Search</h1>
            
            <div className="search-container">
                <input
                    type="text"
                    className="search-input"
                    placeholder="검색어를 입력하세요..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    onKeyPress={handleKeyPress}
                />
                <button
                    className="search-button"
                    onClick={searchVideos}
                    disabled={loading}
                >
                    {loading ? '검색 중...' : '검색'}
                </button>
            </div>

            {error && <div className="error">{error}</div>}

            <div className="video-grid">
                {videos.map(video => (
                    <div
                        key={video.id}
                        className="video-card"
                        onClick={() => openVideo(video.id)}
                    >
                        <img
                            src={video.thumbnail}
                            alt={video.title}
                            className="video-thumbnail"
                        />
                        <div className="video-info">
                            <div className="video-title">{video.title}</div>
                            <div className="video-channel">{video.channel}</div>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
}

ReactDOM.render(<App />, document.getElementById('root'));