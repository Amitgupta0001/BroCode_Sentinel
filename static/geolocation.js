// Geolocation Module
// Tracks user location for security monitoring

class GeoLocation {
    constructor() {
        this.location = null;
        this.ipLocation = null;
    }

    async collect() {
        // Collect both browser geolocation and IP-based location
        const [browserLoc, ipLoc] = await Promise.all([
            this.getBrowserLocation(),
            this.getIPLocation()
        ]);

        this.location = browserLoc;
        this.ipLocation = ipLoc;

        return {
            browser: browserLoc,
            ip: ipLoc,
            timestamp: Date.now()
        };
    }

    async getBrowserLocation() {
        return new Promise((resolve) => {
            if (!navigator.geolocation) {
                resolve({ error: 'geolocation_not_supported' });
                return;
            }

            navigator.geolocation.getCurrentPosition(
                (position) => {
                    resolve({
                        latitude: position.coords.latitude,
                        longitude: position.coords.longitude,
                        accuracy: position.coords.accuracy,
                        altitude: position.coords.altitude,
                        heading: position.coords.heading,
                        speed: position.coords.speed
                    });
                },
                (error) => {
                    resolve({
                        error: error.code === 1 ? 'permission_denied' :
                            error.code === 2 ? 'position_unavailable' :
                                'timeout'
                    });
                },
                {
                    enableHighAccuracy: true,
                    timeout: 5000,
                    maximumAge: 0
                }
            );
        });
    }

    async getIPLocation() {
        try {
            // Using ipapi.co free tier (no API key needed, 1000 requests/day)
            const response = await fetch('https://ipapi.co/json/');
            if (!response.ok) throw new Error('IP location failed');

            const data = await response.json();
            return {
                ip: data.ip,
                city: data.city,
                region: data.region,
                country: data.country_name,
                country_code: data.country_code,
                latitude: data.latitude,
                longitude: data.longitude,
                timezone: data.timezone,
                isp: data.org
            };
        } catch (err) {
            console.error('IP location error:', err);
            return { error: 'ip_location_failed' };
        }
    }

    calculateDistance(lat1, lon1, lat2, lon2) {
        // Haversine formula to calculate distance between two points
        const R = 6371; // Earth's radius in km
        const dLat = this.toRad(lat2 - lat1);
        const dLon = this.toRad(lon2 - lon1);

        const a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
            Math.cos(this.toRad(lat1)) * Math.cos(this.toRad(lat2)) *
            Math.sin(dLon / 2) * Math.sin(dLon / 2);

        const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
        return R * c; // Distance in km
    }

    toRad(degrees) {
        return degrees * (Math.PI / 180);
    }

    isImpossibleTravel(prevLat, prevLon, prevTime, currLat, currLon, currTime) {
        // Calculate if travel between two locations in given time is physically possible
        const distance = this.calculateDistance(prevLat, prevLon, currLat, currLon);
        const timeDiff = (currTime - prevTime) / (1000 * 60 * 60); // hours

        // Assume max speed of 900 km/h (commercial flight)
        const maxSpeed = 900;
        const requiredSpeed = distance / timeDiff;

        return {
            impossible: requiredSpeed > maxSpeed,
            distance_km: Math.round(distance),
            time_hours: Math.round(timeDiff * 10) / 10,
            required_speed_kmh: Math.round(requiredSpeed),
            max_speed_kmh: maxSpeed
        };
    }
}

// Export for use in other modules
window.GeoLocation = GeoLocation;
