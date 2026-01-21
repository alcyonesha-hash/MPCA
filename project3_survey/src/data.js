/**
 * Survey Data - 10 Comparison Sets
 *
 * Structure:
 * - Sets 1-4: Full vs Baseline (text)
 * - Sets 5-6: Full vs (Full - Timing) (GIF)
 * - Sets 7-8: Full vs (Full - Topical Fit) (text)
 * - Sets 9-10: Full vs (Full - Chunking) (GIF)
 *
 * Each set has:
 * - full: The complete human-like metrics version
 * - comparison: The ablated/baseline version
 * - type: "text" or "gif"
 * - comparisonType: What's being compared against Full
 */

const SURVEY_SETS = [
    // ============================================
    // Sets 1-4: Full vs Baseline
    // ============================================

    // Set 1: apt-get package issue
    {
        type: "text",
        comparisonType: "Baseline",
        topic: "apt-get package installation",
        full: [
            { role: "user", sender: "marcopolo", text: "apt-get says package not found when I try to install chromium", time: "14:23" },
            { role: "agent", sender: "helper", text: "Try running apt update first.", time: "14:23 (+5s)" },
            { role: "user", sender: "marcopolo", text: "still says package not found", time: "14:24" },
            { role: "agent", sender: "helper", text: "Chromium is snap-only on Ubuntu now.", time: "14:24 (+7s)" },
            { role: "agent", sender: "helper", text: "Use: snap install chromium", time: "14:24 (+2s)" },
            { role: "user", sender: "marcopolo", text: "oh that explains it, thanks!", time: "14:25" },
            { role: "agent", sender: "helper", text: "No problem.", time: "14:25 (+4s)" },
        ],
        comparison: [
            { role: "user", sender: "marcopolo", text: "apt-get says package not found when I try to install chromium", time: "14:23" },
            { role: "agent", sender: "helper", text: "You should run sudo apt update first to refresh your package lists, then try installing again. If that doesn't work, note that Chromium browser is now only available as a snap package on Ubuntu, not as a deb. You can install it using snap install chromium instead.", time: "14:23 (+0s)" },
            { role: "user", sender: "marcopolo", text: "still says package not found", time: "14:24" },
            { role: "agent", sender: "helper", text: "As I mentioned, Chromium is no longer available via apt on Ubuntu. The deb package is just a transitional package that points to snap. Please use: sudo snap install chromium", time: "14:24 (+0s)" },
            { role: "user", sender: "marcopolo", text: "oh that explains it, thanks!", time: "14:25" },
            { role: "agent", sender: "helper", text: "You're welcome! Happy to help. Let me know if you have any other questions about package management on Ubuntu.", time: "14:25 (+0s)" },
        ]
    },

    // Set 2: cron job issue
    {
        type: "text",
        comparisonType: "Baseline",
        topic: "cron job not running",
        full: [
            { role: "user", sender: "Gudddu", text: "I feel that my cron is not working. */5 * * * /home/ubuntu/script.sh", time: "04:31" },
            { role: "agent", sender: "helper", text: "Missing asterisk after */5.", time: "04:31 (+6s)" },
            { role: "agent", sender: "helper", text: "Should be: */5 * * * *", time: "04:31 (+2s)" },
            { role: "user", sender: "Gudddu", text: "oh I see, let me try", time: "04:32" },
            { role: "user", sender: "Gudddu", text: "Got it. I was missing the right syntax.", time: "04:38" },
            { role: "agent", sender: "helper", text: "Great!", time: "04:38 (+3s)" },
        ],
        comparison: [
            { role: "user", sender: "Gudddu", text: "I feel that my cron is not working. */5 * * * /home/ubuntu/script.sh", time: "04:31" },
            { role: "agent", sender: "helper", text: "I can see the issue with your cron syntax. You have */5 * * * /home/ubuntu/script.sh but the correct format requires 5 fields before the command. The format is: minute hour day month weekday command. You're missing one asterisk - it should be */5 * * * * /home/ubuntu/script.sh (note the 5 asterisks/time fields). Also make sure your script is executable with chmod +x.", time: "04:31 (+0s)" },
            { role: "user", sender: "Gudddu", text: "oh I see, let me try", time: "04:32" },
            { role: "user", sender: "Gudddu", text: "Got it. I was missing the right syntax.", time: "04:38" },
            { role: "agent", sender: "helper", text: "Excellent! Glad that fixed it. Cron syntax can be tricky - I recommend using crontab.guru website to verify your cron expressions in the future.", time: "04:38 (+0s)" },
        ]
    },

    // Set 3: nano editor line wrapping
    {
        type: "text",
        comparisonType: "Baseline",
        topic: "nano editor configuration",
        full: [
            { role: "user", sender: "charles_", text: "Does anyone know how to cause Nano to show the whole line of text?", time: "15:33" },
            { role: "agent", sender: "helper", text: "Use nano -S for soft wrap.", time: "15:33 (+8s)" },
            { role: "user", sender: "charles_", text: "thanks!", time: "15:36" },
            { role: "agent", sender: "helper", text: "Be careful with config files though.", time: "15:36 (+5s)" },
            { role: "agent", sender: "helper", text: "Soft wrap can break them when copying.", time: "15:37 (+2s)" },
            { role: "user", sender: "charles_", text: "interesting, good to know", time: "15:44" },
        ],
        comparison: [
            { role: "user", sender: "charles_", text: "Does anyone know how to cause Nano to show the whole line of text?", time: "15:33" },
            { role: "agent", sender: "helper", text: "To enable line wrapping in nano, you can use the -S flag for soft wrapping when opening a file (nano -S filename), or you can set it permanently by adding 'set softwrap' to your ~/.nanorc configuration file. You can also toggle it while in nano using Meta+S (usually Alt+S). Note that there's also hard wrapping with -w flag which is different.", time: "15:33 (+0s)" },
            { role: "user", sender: "charles_", text: "thanks!", time: "15:36" },
            { role: "agent", sender: "helper", text: "You're welcome! One important thing to be aware of: when using soft wrap, be careful when editing configuration files because the visual line breaks might cause issues when you copy and paste text using mouse selection. The wrapped lines will be treated as separate lines. When saving, it's fine because nano knows the actual line boundaries, but copying can introduce unwanted newlines.", time: "15:36 (+0s)" },
            { role: "user", sender: "charles_", text: "interesting, good to know", time: "15:44" },
        ]
    },

    // Set 4: fsck filesystem repair
    {
        type: "text",
        comparisonType: "Baseline",
        topic: "filesystem check and repair",
        full: [
            { role: "user", sender: "foo", text: "My laptop keeps dropping to maintenance mode. How do I force filesystem check?", time: "20:24" },
            { role: "agent", sender: "helper", text: "Run fsck /dev/sdXY from recovery.", time: "20:24 (+7s)" },
            { role: "user", sender: "foo", text: "which partition should I check?", time: "20:25" },
            { role: "agent", sender: "helper", text: "Usually sda2 for root.", time: "20:25 (+5s)" },
            { role: "agent", sender: "helper", text: "Check with lsblk first.", time: "20:25 (+2s)" },
            { role: "user", sender: "foo", text: "ok will try that", time: "20:26" },
            { role: "agent", sender: "helper", text: "Back up data first if possible.", time: "20:26 (+4s)" },
        ],
        comparison: [
            { role: "user", sender: "foo", text: "My laptop keeps dropping to maintenance mode. How do I force filesystem check?", time: "20:24" },
            { role: "agent", sender: "helper", text: "To force a filesystem check, you can boot into recovery mode or use a live USB. From there, run 'fsck /dev/sdXY' where X is the drive letter and Y is the partition number. For example, fsck /dev/sda2 for the second partition on the first drive. You can also add fsck.mode=force fsck.repair=yes to your kernel boot parameters in GRUB by pressing 'e' at the boot menu and editing the linux line.", time: "20:24 (+0s)" },
            { role: "user", sender: "foo", text: "which partition should I check?", time: "20:25" },
            { role: "agent", sender: "helper", text: "The root partition is typically sda2 on most systems (sda1 is usually the EFI or boot partition). You can verify this by running 'lsblk' or 'fdisk -l' to see your partition layout. Look for the partition mounted at '/' or the largest ext4 partition. Be careful with fsck --repair as it can potentially cause data loss if the filesystem is severely corrupted.", time: "20:25 (+0s)" },
            { role: "user", sender: "foo", text: "ok will try that", time: "20:26" },
            { role: "agent", sender: "helper", text: "Good luck! I strongly recommend backing up any important data before running repair operations, if at all possible. You might also want to check the SMART status of your drive with 'sudo smartctl -a /dev/sda' to see if there are any hardware issues.", time: "20:26 (+0s)" },
        ]
    },

    // ============================================
    // Sets 5-6: Full vs No Timing (GIF)
    // Shows real-time message appearance
    // ============================================

    // Set 5: Quick help exchange - timing difference
    {
        type: "gif",
        comparisonType: "NoTiming",
        topic: "quick help exchange",
        full: {
            src: "data/gifs/timing_full_1.gif",
            description: "Messages appear with natural delays (3-10 seconds)"
        },
        comparison: {
            src: "data/gifs/timing_notiming_1.gif",
            description: "Messages appear instantly (0 second delay)"
        }
    },

    // Set 6: Technical question - timing difference
    {
        type: "gif",
        comparisonType: "NoTiming",
        topic: "technical question",
        full: {
            src: "data/gifs/timing_full_2.gif",
            description: "Messages appear with natural delays"
        },
        comparison: {
            src: "data/gifs/timing_notiming_2.gif",
            description: "Messages appear instantly"
        }
    },

    // ============================================
    // Sets 7-8: Full vs No Topical Fit
    // Agent responds to wrong topic / off-topic
    // ============================================

    // Set 7: Two interleaved conversations - topical fit difference
    {
        type: "text",
        comparisonType: "NoTopicalFit",
        topic: "interleaved conversations",
        full: [
            { role: "user", sender: "alice", text: "How do I check disk space on Ubuntu?", time: "10:01" },
            { role: "user", sender: "bob", text: "Anyone know a good video editor for Linux?", time: "10:01" },
            { role: "agent", sender: "helper", text: "alice: Use df -h command.", time: "10:01 (+6s)" },
            { role: "user", sender: "alice", text: "thanks! what about inode usage?", time: "10:02" },
            { role: "agent", sender: "helper", text: "df -i shows inode usage.", time: "10:02 (+5s)" },
            { role: "user", sender: "bob", text: "looking for something like premiere", time: "10:03" },
            { role: "agent", sender: "helper", text: "bob: Try Kdenlive or DaVinci Resolve.", time: "10:03 (+7s)" },
        ],
        comparison: [
            { role: "user", sender: "alice", text: "How do I check disk space on Ubuntu?", time: "10:01" },
            { role: "user", sender: "bob", text: "Anyone know a good video editor for Linux?", time: "10:01" },
            { role: "agent", sender: "helper", text: "Try Kdenlive, it's a great free video editor!", time: "10:01 (+0s)" },
            { role: "user", sender: "alice", text: "thanks! what about inode usage?", time: "10:02" },
            { role: "agent", sender: "helper", text: "Kdenlive has good timeline features.", time: "10:02 (+0s)" },
            { role: "user", sender: "bob", text: "looking for something like premiere", time: "10:03" },
            { role: "agent", sender: "helper", text: "DaVinci Resolve is more professional.", time: "10:03 (+0s)" },
        ]
    },

    // Set 8: Topic drift - agent stays on vs drifts
    {
        type: "text",
        comparisonType: "NoTopicalFit",
        topic: "wifi troubleshooting",
        full: [
            { role: "user", sender: "user1", text: "My wifi keeps disconnecting after suspend", time: "11:10" },
            { role: "agent", sender: "helper", text: "Check your network manager logs.", time: "11:10 (+6s)" },
            { role: "user", sender: "user1", text: "where are those logs?", time: "11:11" },
            { role: "agent", sender: "helper", text: "journalctl -u NetworkManager", time: "11:11 (+5s)" },
            { role: "user", sender: "user2", text: "anyone tried the new kernel 6.5?", time: "11:12" },
            { role: "user", sender: "user1", text: "I see some errors about wlp2s0", time: "11:13" },
            { role: "agent", sender: "helper", text: "user1: That's your wifi interface.", time: "11:13 (+4s)" },
            { role: "agent", sender: "helper", text: "Try: sudo systemctl restart NetworkManager", time: "11:13 (+2s)" },
        ],
        comparison: [
            { role: "user", sender: "user1", text: "My wifi keeps disconnecting after suspend", time: "11:10" },
            { role: "agent", sender: "helper", text: "Check your network manager logs.", time: "11:10 (+0s)" },
            { role: "user", sender: "user1", text: "where are those logs?", time: "11:11" },
            { role: "agent", sender: "helper", text: "journalctl -u NetworkManager", time: "11:11 (+0s)" },
            { role: "user", sender: "user2", text: "anyone tried the new kernel 6.5?", time: "11:12" },
            { role: "agent", sender: "helper", text: "Yes! Kernel 6.5 has great improvements.", time: "11:12 (+0s)" },
            { role: "user", sender: "user1", text: "I see some errors about wlp2s0", time: "11:13" },
            { role: "agent", sender: "helper", text: "The new kernel has better hardware support overall. You might want to upgrade!", time: "11:13 (+0s)" },
        ]
    },

    // ============================================
    // Sets 9-10: Full vs No Chunking (GIF)
    // Shows message splitting vs single long message
    // ============================================

    // Set 9: Long technical answer - chunking difference
    {
        type: "gif",
        comparisonType: "NoChunking",
        topic: "nginx configuration",
        full: {
            src: "data/gifs/chunking_full_1.gif",
            description: "Response split into natural chunks with typing delays"
        },
        comparison: {
            src: "data/gifs/chunking_nochunk_1.gif",
            description: "Single long message appears all at once"
        }
    },

    // Set 10: Multi-step instruction - chunking difference
    {
        type: "gif",
        comparisonType: "NoChunking",
        topic: "system update steps",
        full: {
            src: "data/gifs/chunking_full_2.gif",
            description: "Instructions appear step by step"
        },
        comparison: {
            src: "data/gifs/chunking_nochunk_2.gif",
            description: "All instructions in one message"
        }
    }
];

// Export for use in app.js
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { SURVEY_SETS };
}
