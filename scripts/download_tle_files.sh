#!/bin/bash
# Download real TLE files for Starlink and OneWeb from Celestrak

mkdir -p data/tle

echo "Downloading Starlink Shell 1 TLEs from Celestrak..."
curl -s "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle" > data/tle/starlink_shell1.tle

echo "Downloading OneWeb TLEs from Celestrak..."
curl -s "https://celestrak.org/NORAD/elements/gp.php?GROUP=oneweb&FORMAT=tle" > data/tle/oneweb.tle

echo "✓ TLE files downloaded to data/tle/"
echo "  - starlink_shell1.tle"
echo "  - oneweb.tle"

# Count satellites
echo ""
echo "Satellite counts:"
echo "  Starlink: $(grep -c "^STARLINK" data/tle/starlink_shell1.tle 2>/dev/null || echo "0")"
echo "  OneWeb: $(grep -c "^ONEWEB" data/tle/oneweb.tle 2>/dev/null || echo "0")"

