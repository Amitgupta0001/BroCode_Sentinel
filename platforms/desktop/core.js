// BroCode Sentinel - Desktop App Main Process (Electron)
// Handles system integration, local storage, and secure window management

const { app, BrowserWindow, ipcMain, Tray, Menu, Notification } = require('electron');
const path = require('path');
const axios = require('axios');
const fs = require('fs');

let mainWindow;
let tray;
let isQuitting = false;

// API Configuration
const API_BASE_URL = 'http://localhost:5000';
let userSession = null;

function createWindow() {
    mainWindow = new BrowserWindow({
        width: 1200,
        height: 800,
        title: "BroCode Sentinel Desktop",
        icon: path.join(__dirname, 'assets/icon.png'),
        webPreferences: {
            nodeIntegration: false,
            contextIsolation: true,
            preload: path.join(__dirname, 'preload.js')
        },
        show: false,
        backgroundColor: '#0f172a' // Dark slate for premium feel
    });

    mainWindow.loadFile('index.html');

    mainWindow.once('ready-to-show', () => {
        mainWindow.show();
    });

    mainWindow.on('close', (event) => {
        if (!isQuitting) {
            event.preventDefault();
            mainWindow.hide();
        }
    });
}

function createTray() {
    tray = new Tray(path.join(__dirname, 'assets/icon_small.png'));
    const contextMenu = Menu.buildFromTemplate([
        { label: 'Show Dashboard', click: () => mainWindow.show() },
        { label: 'Security Status: Active', enabled: false },
        { type: 'separator' },
        { label: 'Pause Monitoring', click: () => toggleMonitoring(false) },
        { label: 'Resume Monitoring', click: () => toggleMonitoring(true) },
        { type: 'separator' },
        { label: 'Settings', click: () => openSettings() },
        {
            label: 'Quit', click: () => {
                isQuitting = true;
                app.quit();
            }
        }
    ]);

    tray.setToolTip('BroCode Sentinel - Protected');
    tray.setContextMenu(contextMenu);

    tray.on('double-click', () => {
        mainWindow.show();
    });
}

// IPC Handlers for communication with the renderer process
ipcMain.handle('auth:login', async (event, credentials) => {
    try {
        const response = await axios.post(`${API_BASE_URL}/authenticate`, credentials);
        userSession = response.data;

        // Notify user of successful secure login
        new Notification({
            title: 'Sentinel Active',
            body: `Welcome back, ${credentials.username}. Continuous protection is enabled.`
        }).show();

        return { success: true, data: response.data };
    } catch (error) {
        return { success: false, error: error.message };
    }
});

ipcMain.on('monitor:update-trust', (event, trustScore) => {
    // Update tray icon or color based on trust level
    if (trustScore < 0.3) {
        tray.setImage(path.join(__dirname, 'assets/icon_alert.png'));
        tray.setToolTip('CRITICAL RISK - ACTION REQUIRED');
    } else if (trustScore < 0.6) {
        tray.setImage(path.join(__dirname, 'assets/icon_warning.png'));
    } else {
        tray.setImage(path.join(__dirname, 'assets/icon_small.png'));
    }
});

// System level monitoring integration
function toggleMonitoring(active) {
    mainWindow.webContents.send('monitoring:state-change', active);
}

function openSettings() {
    mainWindow.show();
    mainWindow.webContents.send('navigation:go-to', 'settings');
}

app.whenReady().then(() => {
    createWindow();
    createTray();

    app.on('activate', () => {
        if (BrowserWindow.getAllWindows().length === 0) createWindow();
    });
});

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') app.quit();
});
