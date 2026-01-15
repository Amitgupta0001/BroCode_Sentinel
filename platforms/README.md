# 🛡️ BroCode Sentinel - Browser Extension

**Enterprise-Grade Security Monitoring for Your Browser**

Monitor your authentication trust score, detect security threats, and protect sensitive data directly from your browser.

---

## 🎯 Features

### ✅ Real-Time Trust Score Monitoring
- Live trust score display in extension badge
- Color-coded indicators (Green/Orange/Red)
- Automatic updates every minute
- Session age tracking

### ✅ Sensitive Data Protection
- Warns before submitting forms with low trust score
- Monitors clipboard for sensitive data
- Detects password fields automatically
- Blocks risky actions when trust is low

### ✅ Security Alerts
- Browser notifications for trust drops
- Warnings when accessing sensitive sites
- Session expiration alerts
- Real-time threat detection

### ✅ Smart Monitoring
- Tracks active tabs
- Identifies sensitive websites (banking, admin, etc.)
- Automatic security checks
- Session management

---

## 📦 Installation

### Chrome/Edge

1. **Download Extension**
   ```
   browser_extension/
   ├── manifest.json
   ├── background.js
   ├── popup.html
   ├── popup.js
   ├── content.js
   └── icons/
   ```

2. **Load Extension**
   - Open Chrome/Edge
   - Go to `chrome://extensions/`
   - Enable "Developer mode"
   - Click "Load unpacked"
   - Select the `browser_extension` folder

3. **Configure**
   - Click the extension icon
   - Enter your username
   - Enter your API key
   - Click "Login"

### Firefox

1. **Prepare Extension**
   - Update `manifest.json` to version 2 format (if needed)
   - Or use the same manifest (Firefox supports v3)

2. **Load Extension**
   - Open Firefox
   - Go to `about:debugging#/runtime/this-firefox`
   - Click "Load Temporary Add-on"
   - Select `manifest.json` from `browser_extension` folder

---

## 🚀 Usage

### Initial Setup

1. **Get API Key**
   ```python
   from api_integration import APIKeyManager
   
   api_manager = APIKeyManager()
   api_key = api_manager.generate_api_key("Browser Extension", ["read"])
   print(f"Your API Key: {api_key}")
   ```

2. **Login to Extension**
   - Click extension icon
   - Enter username
   - Enter API key
   - Click "Login"

3. **Monitor Trust Score**
   - Trust score appears in badge
   - Click icon for detailed view
   - Auto-updates every minute

### Features in Action

#### Trust Score Display
```
Badge: 85 (Green)
Popup: 
  - Trust Score: 85%
  - Level: Excellent
  - Last Check: 2 mins ago
  - Session Age: 1h 23m
```

#### Security Warnings
```
⚠️ Security Warning
You are accessing a sensitive site with a low trust score.
Please re-authenticate.
```

#### Form Protection
```
⚠️ BroCode Sentinel Security Warning

Your current trust score is 45%.
Submitting sensitive data with a low trust score is not recommended.

Do you want to proceed anyway?
[Cancel] [OK]
```

---

## 🔧 Configuration

### API Endpoint

Edit `background.js`:
```javascript
const CONFIG = {
  apiUrl: 'https://your-domain.com', // Change this
  checkInterval: 60000, // 1 minute
  sessionTimeout: 3600000 // 1 hour
};
```

### Sensitive Keywords

Edit `content.js`:
```javascript
const SENSITIVE_KEYWORDS = [
  'password', 'credit card', 'ssn',
  'bank account', 'routing number', 'cvv'
];
```

### Trust Thresholds

Edit `background.js`:
```javascript
// Warning threshold
if (trustScore < 0.5) {
  showNotification('Security Alert', ...);
}

// Sensitive site threshold
if (trustScore < 0.7) {
  showNotification('Security Warning', ...);
}
```

---

## 📊 Features Breakdown

### Background Service Worker (`background.js`)
- ✅ API communication
- ✅ Trust score monitoring
- ✅ Session management
- ✅ Notifications
- ✅ Badge updates
- ✅ Tab monitoring
- ✅ Auto-logout

### Popup UI (`popup.html` + `popup.js`)
- ✅ Login/logout
- ✅ Trust score display
- ✅ Session info
- ✅ Manual refresh
- ✅ Settings access
- ✅ Beautiful glassmorphism design

### Content Script (`content.js`)
- ✅ Form monitoring
- ✅ Sensitive data detection
- ✅ Clipboard protection
- ✅ Password field detection
- ✅ Security indicator
- ✅ Real-time warnings

---

## 🎨 UI Screenshots

### Popup - Logged In
```
┌─────────────────────────┐
│   🛡️ BroCode Sentinel   │
│  Enterprise Security    │
├─────────────────────────┤
│  👤 john_doe            │
│     Active              │
├─────────────────────────┤
│     Trust Score         │
│        85%              │
│     [Excellent]         │
├─────────────────────────┤
│  2 mins ago  │  1h 23m  │
│  Last Check  │ Session  │
├─────────────────────────┤
│  [🔄 Refresh Score]     │
│  [⚙️ Settings]          │
│  [🚪 Logout]            │
└─────────────────────────┘
```

### Badge Indicators
- **Green (85)**: Trust ≥ 80% - Excellent
- **Blue (65)**: Trust ≥ 60% - Good
- **Orange (45)**: Trust ≥ 40% - Warning
- **Red (25)**: Trust < 40% - Critical

---

## 🔒 Security Features

### 1. Trust Score Monitoring
- Real-time updates
- Visual indicators
- Automatic alerts

### 2. Sensitive Site Detection
- Banking websites
- Admin panels
- Payment processors
- Account dashboards

### 3. Form Protection
- Password field detection
- Sensitive data warnings
- Low-trust blocking

### 4. Clipboard Monitoring
- Credit card detection
- SSN detection
- Bank account detection

### 5. Session Management
- Auto-logout after timeout
- Session age tracking
- Expiration warnings

---

## 📝 API Integration

The extension uses the BroCode Sentinel API:

```javascript
// Get trust score
GET /api/v1/users/{username}/trust
Headers: X-API-Key: bk_your_api_key

Response:
{
  "username": "john",
  "trust_score": 0.85,
  "components": {...},
  "timestamp": 1234567890
}
```

---

## 🐛 Troubleshooting

### Extension Not Loading
- Check manifest.json syntax
- Ensure all files are present
- Check browser console for errors

### API Connection Failed
- Verify API URL in background.js
- Check API key is valid
- Ensure CORS is enabled on server

### Trust Score Not Updating
- Check API key permissions
- Verify username is correct
- Check network tab for errors

### Badge Not Showing
- Reload extension
- Check background service worker
- Verify permissions in manifest

---

## 🚀 Development

### Build for Production

1. **Update manifest.json**
   ```json
   {
     "host_permissions": [
       "https://your-production-domain.com/*"
     ]
   }
   ```

2. **Update background.js**
   ```javascript
   const CONFIG = {
     apiUrl: 'https://your-production-domain.com'
   };
   ```

3. **Create icons**
   - icon16.png (16x16)
   - icon48.png (48x48)
   - icon128.png (128x128)

4. **Package extension**
   ```bash
   zip -r brocode-sentinel.zip browser_extension/
   ```

### Publish to Chrome Web Store

1. Create developer account
2. Upload ZIP file
3. Fill in store listing
4. Submit for review

---

## 📊 Statistics

- **Files**: 5
- **Lines of Code**: ~800
- **Features**: 10+
- **Supported Browsers**: Chrome, Edge, Firefox
- **API Calls**: Optimized (1/minute)

---

## 🎯 Roadmap

- [ ] Firefox-specific optimizations
- [ ] Safari support
- [ ] Offline mode
- [ ] Advanced analytics
- [ ] Custom themes
- [ ] Keyboard shortcuts

---

## 📞 Support

- **Documentation**: See main README.md
- **API Docs**: See api_integration.py
- **Issues**: GitHub Issues

---

## 📄 License

MIT License - Same as main project

---

**Built with ❤️ for BroCode Sentinel**

**Version**: 1.0.0  
**Status**: ✅ Production Ready  
**Last Updated**: January 12, 2026
