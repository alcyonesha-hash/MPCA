/**
 * Survey Data - 10 Comparison Sets
 *
 * REAL IRC conversations extracted from Ubuntu IRC channel
 * Multi-party conversations with interleaved threads
 *
 * Structure:
 * - Sets 1-4: Full vs Baseline (text)
 * - Sets 5-6: Full vs (Full - Timing) (GIF)
 * - Sets 7-8: Full vs (Full - Topical Fit) (text)
 * - Sets 9-10: Full vs (Full - Chunking) (GIF)
 */

const SURVEY_SETS = [
    // ============================================
    // Sets 1-4: Full vs Baseline
    // ============================================

    // Set 1: Real IRC - apt/snap chromium discussion + webchat samba issue
    // Source: Lines 14-26, 206-209 from ubuntu_merged.txt
    {
        type: "text",
        comparisonType: "Baseline",
        topic: "package installation and file sharing",
        full: [
            { role: "user", sender: "marcopolo", text: "Guys, can you make apt show deb packages instead of snaps?", time: "06:49" },
            { role: "user", sender: "webchat47", text: "I get this error when I try and allow others to create and delete files: net usershare add: share name contains invalid characters", time: "06:50" },
            { role: "user", sender: "ravage", text: "apt only shows apt", time: "06:50" },
            { role: "user", sender: "marcopolo", text: "it seems like apt repositories only show snaps", time: "06:50" },
            { role: "agent", sender: "helper", text: "marcopolo: apt doesn't show snaps", time: "06:51 (+3s)" },
            { role: "agent", sender: "helper", text: "What package are you looking for?", time: "06:51 (+2s)" },
            { role: "user", sender: "marcopolo", text: "In software center there are only snaps packages", time: "06:51" },
            { role: "user", sender: "marcopolo", text: "it says transitional package to snap", time: "06:53" },
            { role: "agent", sender: "helper", text: "webchat47: The error means the share name has invalid chars", time: "06:53 (+4s)" },
            { role: "agent", sender: "helper", text: "Don't use / in the share name", time: "06:53 (+2s)" },
            { role: "user", sender: "ravage", text: "that is probably firefox then", time: "06:54" },
            { role: "user", sender: "marcopolo", text: "no its chromium", time: "06:54" },
            { role: "agent", sender: "helper", text: "marcopolo: Ubuntu doesn't provide deb versions of chromium anymore", time: "06:54 (+3s)" },
            { role: "agent", sender: "helper", text: "It's snap only. Same for firefox.", time: "06:54 (+2s)" },
        ],
        comparison: [
            { role: "user", sender: "marcopolo", text: "Guys, can you make apt show deb packages instead of snaps?", time: "06:49" },
            { role: "user", sender: "webchat47", text: "I get this error when I try and allow others to create and delete files: net usershare add: share name contains invalid characters", time: "06:50" },
            { role: "user", sender: "ravage", text: "apt only shows apt", time: "06:50" },
            { role: "user", sender: "marcopolo", text: "it seems like apt repositories only show snaps", time: "06:50" },
            { role: "agent", sender: "helper", text: "Let me address both questions. For the apt/snap issue: apt doesn't show snaps at all - they're separate package managers. When you see 'transitional package to snap' for certain packages like Chromium or Firefox, it means Ubuntu has decided to distribute those applications exclusively as snaps. There's no deb version available in the official repos anymore. You can either use the snap versions, install from a third-party PPA (not recommended), or use flatpak alternatives. For the samba share error: the message 'share name contains invalid characters' means you're using characters that aren't allowed in share names. Characters like %<>*?|/\\+=;:\", are not permitted. Use a simple alphanumeric name without special characters.", time: "06:50 (+0s)" },
            { role: "user", sender: "marcopolo", text: "In software center there are only snaps packages", time: "06:51" },
            { role: "user", sender: "marcopolo", text: "it says transitional package to snap", time: "06:53" },
            { role: "user", sender: "ravage", text: "that is probably firefox then", time: "06:54" },
            { role: "user", sender: "marcopolo", text: "no its chromium", time: "06:54" },
            { role: "agent", sender: "helper", text: "Yes, as I mentioned, both Chromium and Firefox are snap-only in Ubuntu now. This was a decision by Canonical to make updates easier and more secure. If you really need a deb version, you'd have to use unofficial PPAs.", time: "06:54 (+0s)" },
        ]
    },

    // Set 2: Real IRC - Nano text wrapping discussion
    // Source: Lines 70-82 from ubuntu_merged.txt
    {
        type: "text",
        comparisonType: "Baseline",
        topic: "nano editor line wrapping",
        full: [
            { role: "user", sender: "charles_", text: "Does anyone know how to cause the text editor Nano to show the whole line of text, I think they call it scrolling?", time: "15:33" },
            { role: "user", sender: "leftyfb", text: "charles_ https://blog.wplauncher.com/line-wrapping-in-nano-editor/", time: "15:35" },
            { role: "user", sender: "charles_", text: "thanks leftyfb!", time: "15:36" },
            { role: "agent", sender: "helper", text: "charles_: You can use -w flag to disable wrapping", time: "15:37 (+4s)" },
            { role: "agent", sender: "helper", text: "Or set softwrap in nanorc", time: "15:37 (+2s)" },
            { role: "user", sender: "PeGaSuS", text: "still, be careful with line wrapping when copying text, because what's on the next line is copied as a newline", time: "15:38" },
            { role: "user", sender: "charles_", text: "interesting, PeGaSuS", time: "15:44" },
            { role: "user", sender: "PeGaSuS", text: "and be especially careful when editing config files or soft-wrap might break them", time: "15:45" },
            { role: "user", sender: "charles_", text: "usually on Nano I send out blocks by Ctrl+O, do you think scrolled lines might appear as separate lines?", time: "15:46" },
            { role: "agent", sender: "helper", text: "charles_: The wrap is visual only for saving", time: "15:47 (+5s)" },
            { role: "agent", sender: "helper", text: "It affects copy/paste but not the file itself", time: "15:47 (+2s)" },
            { role: "user", sender: "PeGaSuS", text: "I don't think so. I remember it only affected copy/paste when selecting with the mouse", time: "15:50" },
            { role: "user", sender: "PeGaSuS", text: "as far as nano is aware, when you use Ctrl+K/U the file doesn't have lines wrapped", time: "15:53" },
        ],
        comparison: [
            { role: "user", sender: "charles_", text: "Does anyone know how to cause the text editor Nano to show the whole line of text, I think they call it scrolling?", time: "15:33" },
            { role: "user", sender: "leftyfb", text: "charles_ https://blog.wplauncher.com/line-wrapping-in-nano-editor/", time: "15:35" },
            { role: "user", sender: "charles_", text: "thanks leftyfb!", time: "15:36" },
            { role: "agent", sender: "helper", text: "For nano line wrapping/scrolling: You can disable hard wrapping with the -w flag when starting nano ('nano -w filename'), or enable soft wrapping which allows horizontal scrolling. To make this permanent, add 'set nowrap' or 'set softwrap' to your ~/.nanorc file. Be aware that soft wrapping is purely visual - it doesn't actually add newlines to your file. However, when you copy text from a wrapped display using the mouse and system clipboard (Ctrl+Shift+C), the visual line breaks may be included. When using nano's internal cut/paste (Ctrl+K and Ctrl+U), the wrapping doesn't affect the actual content since nano knows the real line structure.", time: "15:36 (+0s)" },
            { role: "user", sender: "PeGaSuS", text: "still, be careful with line wrapping when copying text, because what's on the next line is copied as a newline", time: "15:38" },
            { role: "user", sender: "charles_", text: "interesting, PeGaSuS", time: "15:44" },
            { role: "user", sender: "PeGaSuS", text: "and be especially careful when editing config files or soft-wrap might break them", time: "15:45" },
            { role: "user", sender: "charles_", text: "usually on Nano I send out blocks by Ctrl+O, do you think scrolled lines might appear as separate lines?", time: "15:46" },
            { role: "user", sender: "PeGaSuS", text: "I don't think so. I remember it only affected copy/paste when selecting with the mouse", time: "15:50" },
            { role: "user", sender: "PeGaSuS", text: "as far as nano is aware, when you use Ctrl+K/U the file doesn't have lines wrapped", time: "15:53" },
            { role: "agent", sender: "helper", text: "Exactly as mentioned earlier, the wrapping is visual only. Nano's internal operations work on the actual file content, not the displayed representation.", time: "15:53 (+0s)" },
        ]
    },

    // Set 3: Real IRC - DHCP/DNS domain search + network speed issues
    // Source: Lines 87-103, 432-448 from ubuntu_merged.txt
    {
        type: "text",
        comparisonType: "Baseline",
        topic: "network configuration issues",
        full: [
            { role: "user", sender: "Tabmow", text: "My DHCP server is set with a domain search that works fine on Windows, but doesn't seem to work on my ubuntu machine", time: "17:10" },
            { role: "user", sender: "Guest24", text: "just need to know if settings/network/wired shows 100mbs does this mean my network driver will not connect higher speed?", time: "17:11" },
            { role: "user", sender: "rbox", text: "Tabmow: what doesn't work?", time: "17:13" },
            { role: "user", sender: "Tabmow", text: "My local domain - if I ping blah.local it works, but if I ping blah it does not", time: "17:18" },
            { role: "agent", sender: "helper", text: "Guest24: That shows the negotiated link speed", time: "17:18 (+4s)" },
            { role: "agent", sender: "helper", text: "Could be cable or switch issue", time: "17:18 (+2s)" },
            { role: "user", sender: "rbox", text: ".local is gonna conflict with multicast dns", time: "17:19" },
            { role: "user", sender: "Tabmow", text: "That was an example", time: "17:19" },
            { role: "user", sender: "rbox", text: "what does /etc/resolv.conf say", time: "17:19" },
            { role: "user", sender: "Tabmow", text: "It has 'search .'", time: "17:20" },
            { role: "agent", sender: "helper", text: "Tabmow: Check resolvectl status", time: "17:20 (+3s)" },
            { role: "agent", sender: "helper", text: "It should list dns domain from DHCP", time: "17:20 (+2s)" },
            { role: "user", sender: "Tabmow", text: "resolvectl status mentions the correct DNS but doesn't say anything about search domains", time: "17:22" },
            { role: "user", sender: "mybalzitch", text: "Guest24: Some drivers don't autonegotiate properly so you have to force it to 1000 mbps", time: "17:22" },
        ],
        comparison: [
            { role: "user", sender: "Tabmow", text: "My DHCP server is set with a domain search that works fine on Windows, but doesn't seem to work on my ubuntu machine", time: "17:10" },
            { role: "user", sender: "Guest24", text: "just need to know if settings/network/wired shows 100mbs does this mean my network driver will not connect higher speed?", time: "17:11" },
            { role: "user", sender: "rbox", text: "Tabmow: what doesn't work?", time: "17:13" },
            { role: "user", sender: "Tabmow", text: "My local domain - if I ping blah.local it works, but if I ping blah it does not", time: "17:18" },
            { role: "agent", sender: "helper", text: "Let me address both issues. For the DHCP domain search problem: Ubuntu uses systemd-resolved for DNS, which handles search domains differently than Windows. Check /etc/resolv.conf to see what's configured (it should be a symlink to systemd's stub resolver). Run 'resolvectl status' to see if the search domain is being received from DHCP. If it shows 'search .' that means no search domain is set. Also note that .local domains can conflict with mDNS (Avahi). For the network speed showing 100Mbps: this is the negotiated link speed between your NIC and switch/router. It could indicate a cable issue (bad cable, only 2 pairs connected), switch port problem, or driver autonegotiation issues. Try a different cable or port. Some drivers need manual speed configuration via ethtool.", time: "17:13 (+0s)" },
            { role: "user", sender: "rbox", text: ".local is gonna conflict with multicast dns", time: "17:19" },
            { role: "user", sender: "Tabmow", text: "That was an example", time: "17:19" },
            { role: "user", sender: "rbox", text: "what does /etc/resolv.conf say", time: "17:19" },
            { role: "user", sender: "Tabmow", text: "It has 'search .'", time: "17:20" },
            { role: "user", sender: "Tabmow", text: "resolvectl status mentions the correct DNS but doesn't say anything about search domains", time: "17:22" },
            { role: "user", sender: "mybalzitch", text: "Guest24: Some drivers don't autonegotiate properly so you have to force it to 1000 mbps", time: "17:22" },
            { role: "agent", sender: "helper", text: "As I mentioned, use resolvectl status to check. If search domain isn't showing, your DHCP server might not be sending it properly, or NetworkManager isn't processing it correctly.", time: "17:22 (+0s)" },
        ]
    },

    // Set 4: Real IRC - Obsidian installation troubleshooting
    // Source: Lines 204-249 from ubuntu_merged.txt
    {
        type: "text",
        comparisonType: "Baseline",
        topic: "application installation troubleshooting",
        full: [
            { role: "user", sender: "chirpingbird", text: "There's a program I installed a year ago and pinned to my dash bar, but I cannot remember if I installed it via a .deb or through snap", time: "22:16" },
            { role: "user", sender: "chirpingbird", text: "Is there an easy way to find that information?", time: "22:16" },
            { role: "user", sender: "rbox", text: "well just look at whats installed in dpkg and snap", time: "22:17" },
            { role: "agent", sender: "helper", text: "chirpingbird: Check ~/.local/share/applications/", time: "22:18 (+4s)" },
            { role: "agent", sender: "helper", text: "Look for .desktop files with Exec line", time: "22:18 (+2s)" },
            { role: "user", sender: "chirpingbird", text: "I went to the directory and ran grep -rni 'obsidian' but didn't see anything", time: "22:23" },
            { role: "user", sender: "EriC^^", text: "chirpingbird: try dpkg -l | grep obsidian", time: "22:25" },
            { role: "user", sender: "chirpingbird", text: "I did get something for dpkg! It shows: rc obsidian 1.5.3", time: "22:28" },
            { role: "agent", sender: "helper", text: "chirpingbird: 'rc' means removed but config remains", time: "22:29 (+5s)" },
            { role: "user", sender: "EriC^^", text: "that means it's been removed but the config files are still there", time: "22:29" },
            { role: "user", sender: "oerheks", text: "appimage?", time: "22:32" },
            { role: "user", sender: "chirpingbird", text: "snap list | grep obsidian shows empty", time: "22:32" },
            { role: "agent", sender: "helper", text: "oerheks: Good thought. Check ~/bin for AppImage", time: "22:33 (+3s)" },
            { role: "user", sender: "EriC^^", text: "if you want try pressing the dash icon, and with it running look at ps -e", time: "22:35" },
            { role: "user", sender: "chirpingbird", text: "It points to an AppImage! /home/chirpingbird/bin/Obsidian-0.15.6.AppImage", time: "22:38" },
            { role: "user", sender: "chirpingbird", text: "Thank you so much! I have the information I needed!", time: "22:39" },
        ],
        comparison: [
            { role: "user", sender: "chirpingbird", text: "There's a program I installed a year ago and pinned to my dash bar, but I cannot remember if I installed it via a .deb or through snap", time: "22:16" },
            { role: "user", sender: "chirpingbird", text: "Is there an easy way to find that information?", time: "22:16" },
            { role: "user", sender: "rbox", text: "well just look at whats installed in dpkg and snap", time: "22:17" },
            { role: "agent", sender: "helper", text: "To find out how an application was installed, there are several methods: 1) Check dpkg with 'dpkg -l | grep appname' - if it shows 'ii' status, it's installed via apt/deb. If 'rc' status, it was installed but removed (configs remain). 2) Check snap with 'snap list | grep appname'. 3) Check flatpak with 'flatpak list | grep appname'. 4) For AppImages, check your home directory, typically ~/Applications or ~/bin. 5) Desktop files in ~/.local/share/applications/ contain the Exec path which reveals the location. 6) You can also run the app and check 'ps aux | grep appname' to see the process path. The 'which appname' or 'whereis appname' commands can also help locate binaries.", time: "22:17 (+0s)" },
            { role: "user", sender: "chirpingbird", text: "I went to the directory and ran grep -rni 'obsidian' but didn't see anything", time: "22:23" },
            { role: "user", sender: "EriC^^", text: "chirpingbird: try dpkg -l | grep obsidian", time: "22:25" },
            { role: "user", sender: "chirpingbird", text: "I did get something for dpkg! It shows: rc obsidian 1.5.3", time: "22:28" },
            { role: "user", sender: "EriC^^", text: "that means it's been removed but the config files are still there", time: "22:29" },
            { role: "user", sender: "oerheks", text: "appimage?", time: "22:32" },
            { role: "user", sender: "chirpingbird", text: "snap list | grep obsidian shows empty", time: "22:32" },
            { role: "user", sender: "EriC^^", text: "if you want try pressing the dash icon, and with it running look at ps -e", time: "22:35" },
            { role: "user", sender: "chirpingbird", text: "It points to an AppImage! /home/chirpingbird/bin/Obsidian-0.15.6.AppImage", time: "22:38" },
            { role: "user", sender: "chirpingbird", text: "Thank you so much! I have the information I needed!", time: "22:39" },
            { role: "agent", sender: "helper", text: "Great, as I mentioned earlier, AppImages are often stored in home directories. The process of checking running processes is a reliable way to find the actual executable path.", time: "22:39 (+0s)" },
        ]
    },

    // ============================================
    // Sets 5-6: Full vs No Timing (GIF)
    // ============================================

    // Set 5: Real IRC - System suspend/resume + dark theme (interleaved)
    // Source: Lines 286-334 from ubuntu_merged.txt
    {
        type: "gif",
        comparisonType: "NoTiming",
        topic: "system issues troubleshooting",
        full: {
            src: "data/gifs/timing_full_1.gif",
            description: "Messages appear with natural delays"
        },
        comparison: {
            src: "data/gifs/timing_notiming_1.gif",
            description: "Messages appear instantly"
        }
    },

    // Set 6: Real IRC - Terminal ctrl-a and phased updates
    // Source: Lines 450-499, 623-645 from ubuntu_merged.txt
    {
        type: "gif",
        comparisonType: "NoTiming",
        topic: "terminal and package updates",
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
    // ============================================

    // Set 7: Real IRC - VM networking + gsettings keybindings (interleaved threads)
    // Source: Lines 360-412 from ubuntu_merged.txt
    {
        type: "text",
        comparisonType: "NoTopicalFit",
        topic: "VM networking and keybindings",
        full: [
            { role: "user", sender: "alcosta", text: "I have two PCs with Virtual Machines, I want all machines to communicate together", time: "18:29" },
            { role: "user", sender: "Yakov", text: "I just installed sublime from source but there is no preferences. How to fix it?", time: "18:31" },
            { role: "user", sender: "leftyfb", text: "alcosta: which hypervisor?", time: "18:29" },
            { role: "user", sender: "alcosta", text: "qemu/KVM with Virtual Machine Manager GUI", time: "18:30" },
            { role: "agent", sender: "helper", text: "alcosta: Create a bridge in netplan first", time: "18:31 (+4s)" },
            { role: "agent", sender: "helper", text: "Yakov: Seek support from the vendor for that", time: "18:32 (+3s)" },
            { role: "user", sender: "alcosta", text: "At reboot time, the old configuration for internet disappeared!", time: "18:31" },
            { role: "user", sender: "leftyfb", text: "Yakov: you'll have to seek support from the 3rd party vendor", time: "18:32" },
            { role: "user", sender: "Yakov", text: "got it", time: "18:32" },
            { role: "agent", sender: "helper", text: "alcosta: Provide your netplan configs via pastebin", time: "18:33 (+4s)" },
            { role: "user", sender: "leftyfb", text: "alcosta: you should provide netplan configs for the host and guests", time: "18:33" },
            { role: "user", sender: "alcosta", text: "My original yaml has: renderer: NetworkManager", time: "18:36" },
            { role: "agent", sender: "helper", text: "alcosta: So you're using NetworkManager, not netplan directly", time: "18:37 (+3s)" },
            { role: "agent", sender: "helper", text: "Stick with NetworkManager for desktop", time: "18:37 (+2s)" },
        ],
        comparison: [
            { role: "user", sender: "alcosta", text: "I have two PCs with Virtual Machines, I want all machines to communicate together", time: "18:29" },
            { role: "user", sender: "Yakov", text: "I just installed sublime from source but there is no preferences. How to fix it?", time: "18:31" },
            { role: "user", sender: "leftyfb", text: "alcosta: which hypervisor?", time: "18:29" },
            { role: "user", sender: "alcosta", text: "qemu/KVM with Virtual Machine Manager GUI", time: "18:30" },
            { role: "agent", sender: "helper", text: "For Sublime Text preferences, check the Preferences menu", time: "18:31 (+0s)" },
            { role: "user", sender: "alcosta", text: "At reboot time, the old configuration for internet disappeared!", time: "18:31" },
            { role: "user", sender: "leftyfb", text: "Yakov: you'll have to seek support from the 3rd party vendor", time: "18:32" },
            { role: "user", sender: "Yakov", text: "got it", time: "18:32" },
            { role: "agent", sender: "helper", text: "Yes, Sublime is third party software, good choice!", time: "18:33 (+0s)" },
            { role: "user", sender: "leftyfb", text: "alcosta: you should provide netplan configs for the host and guests", time: "18:33" },
            { role: "user", sender: "alcosta", text: "My original yaml has: renderer: NetworkManager", time: "18:36" },
            { role: "agent", sender: "helper", text: "YAML files are great for configuration management", time: "18:37 (+0s)" },
            { role: "agent", sender: "helper", text: "Many editors support YAML syntax highlighting", time: "18:37 (+0s)" },
        ]
    },

    // Set 8: Real IRC - Boot repair + WhatsApp installation (interleaved)
    // Source: Lines 673-702 from ubuntu_merged.txt
    {
        type: "text",
        comparisonType: "NoTopicalFit",
        topic: "boot issues and app installation",
        full: [
            { role: "user", sender: "cahoots", text: "my machine won't boot at all. It just takes me straight to bios settings", time: "16:35" },
            { role: "user", sender: "Urk", text: "Is there a desktop version of whatsapp in the repos?", time: "18:57" },
            { role: "user", sender: "EriC^^", text: "cahoots: can you share 'sudo parted -ls | nc termbin.com 9999'?", time: "17:01" },
            { role: "agent", sender: "helper", text: "Urk: sudo snap install whatsapp-for-linux", time: "18:58 (+3s)" },
            { role: "user", sender: "leftyfb", text: "Urk: sudo snap install whatsapp-app", time: "18:58" },
            { role: "user", sender: "cahoots", text: "here's the contents of parted command: https://termbin.com/6kvy1", time: "17:21" },
            { role: "agent", sender: "helper", text: "cahoots: Check your UEFI boot mode in BIOS", time: "17:22 (+5s)" },
            { role: "agent", sender: "helper", text: "Is CSM legacy disabled?", time: "17:22 (+2s)" },
            { role: "user", sender: "Urk", text: "What do I do now that it is downloaded with snap?", time: "19:00" },
            { role: "user", sender: "leftyfb", text: "run it", time: "19:01" },
            { role: "agent", sender: "helper", text: "Urk: Open it like any other app", time: "19:01 (+3s)" },
            { role: "user", sender: "Urk", text: "Whatsapp prompting me to link my phone with QR code", time: "19:07" },
            { role: "user", sender: "oerheks", text: "yes, you need a phone for that first", time: "19:07" },
            { role: "agent", sender: "helper", text: "cahoots: Try creating the EFI entry manually", time: "19:10 (+4s)" },
        ],
        comparison: [
            { role: "user", sender: "cahoots", text: "my machine won't boot at all. It just takes me straight to bios settings", time: "16:35" },
            { role: "user", sender: "Urk", text: "Is there a desktop version of whatsapp in the repos?", time: "18:57" },
            { role: "user", sender: "EriC^^", text: "cahoots: can you share 'sudo parted -ls | nc termbin.com 9999'?", time: "17:01" },
            { role: "agent", sender: "helper", text: "WhatsApp is a great messaging app!", time: "18:58 (+0s)" },
            { role: "user", sender: "leftyfb", text: "Urk: sudo snap install whatsapp-app", time: "18:58" },
            { role: "user", sender: "cahoots", text: "here's the contents of parted command: https://termbin.com/6kvy1", time: "17:21" },
            { role: "agent", sender: "helper", text: "Snap packages are easy to install and update", time: "17:22 (+0s)" },
            { role: "user", sender: "Urk", text: "What do I do now that it is downloaded with snap?", time: "19:00" },
            { role: "user", sender: "leftyfb", text: "run it", time: "19:01" },
            { role: "agent", sender: "helper", text: "Running apps is straightforward on Ubuntu", time: "19:01 (+0s)" },
            { role: "user", sender: "Urk", text: "Whatsapp prompting me to link my phone with QR code", time: "19:07" },
            { role: "user", sender: "oerheks", text: "yes, you need a phone for that first", time: "19:07" },
            { role: "agent", sender: "helper", text: "QR codes are commonly used for authentication", time: "19:10 (+0s)" },
        ]
    },

    // ============================================
    // Sets 9-10: Full vs No Chunking (GIF)
    // ============================================

    // Set 9: Real IRC - Network bridge setup with multiple users
    // Source: Lines 521-586 from ubuntu_merged.txt
    {
        type: "gif",
        comparisonType: "NoChunking",
        topic: "network bridge configuration",
        full: {
            src: "data/gifs/chunking_full_1.gif",
            description: "Response split into natural chunks"
        },
        comparison: {
            src: "data/gifs/chunking_nochunk_1.gif",
            description: "Single long message"
        }
    },

    // Set 10: Real IRC - GRUB/EFI boot repair steps
    // Source: Lines 716-799 from ubuntu_merged.txt
    {
        type: "gif",
        comparisonType: "NoChunking",
        topic: "GRUB installation steps",
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

// GIF conversation data for generation script
const GIF_CONVERSATIONS = {
    // Set 5: System suspend/resume + dark theme (timing comparison)
    timing_1: {
        messages: [
            { role: "user", sender: "arkanoid", text: "My system fails to resume to gnome desktop after suspend. When it resumes gnome shell is a black screen", ts: 0 },
            { role: "user", sender: "derek-shnosh", text: "In Ubuntu 23.10, some apps are not honoring the dark theme for parts of the window", ts: 2000 },
            { role: "user", sender: "lotuspsychje", text: "arkanoid: can you share your dmesg with the volunteers please", ts: 4000 },
            { role: "agent", sender: "helper", text: "arkanoid: Try loginctl terminate-session", ts: 8000, noTimingTs: 4100 },
            { role: "agent", sender: "helper", text: "This resets the gnome session without reboot", ts: 10500, noTimingTs: 4200 },
            { role: "user", sender: "arkanoid", text: "just tried systemctl restart systemd-logind, got visual login back", ts: 12000 },
            { role: "user", sender: "ioria", text: "derek-shnosh: run in terminal 'G_MESSAGES_DEBUG=all hexchat'", ts: 14000 },
            { role: "agent", sender: "helper", text: "derek-shnosh: Check your gtk-theme setting", ts: 18000, noTimingTs: 14100 },
            { role: "user", sender: "derek-shnosh", text: "Figured it out! gtk-theme was set to Yaru-blue, changed to Yaru-blue-dark", ts: 22000 },
            { role: "agent", sender: "helper", text: "arkanoid: Good, the session restart worked", ts: 25000, noTimingTs: 22100 },
            { role: "agent", sender: "helper", text: "This is likely bug #1968907 in gnome-shell", ts: 27500, noTimingTs: 22200 },
        ]
    },

    // Set 6: Terminal ctrl-a + phased updates (timing comparison)
    timing_2: {
        messages: [
            { role: "user", sender: "en1gma", text: "ctrl-a does not copy all in terminal but works in firefox. why?", ts: 0 },
            { role: "user", sender: "ELFrederich", text: "7 packages can be upgraded but apt upgrade keeps them back", ts: 2000 },
            { role: "user", sender: "pragmaticenigma", text: "ctrl+a doesn't copy anything, it's select all. Terminal doesn't support that command", ts: 4000 },
            { role: "agent", sender: "helper", text: "en1gma: Terminal apps handle shortcuts differently", ts: 8000, noTimingTs: 4100 },
            { role: "agent", sender: "helper", text: "ctrl-a in bash moves cursor to line start", ts: 10500, noTimingTs: 4200 },
            { role: "user", sender: "lotuspsychje", text: "ELFrederich: those are phased updates", ts: 12000 },
            { role: "user", sender: "en1gma", text: "i was reading that /etc/bash.bashrc needs to be edited", ts: 14000 },
            { role: "agent", sender: "helper", text: "ELFrederich: Phased updates roll out gradually", ts: 18000, noTimingTs: 14100 },
            { role: "agent", sender: "helper", text: "Just wait a few days and they'll be available", ts: 20500, noTimingTs: 14200 },
            { role: "user", sender: "pragmaticenigma", text: "en1gma: don't edit that file, it could break your system", ts: 22000 },
            { role: "agent", sender: "helper", text: "en1gma: Edit ~/.bashrc instead, it's safer", ts: 26000, noTimingTs: 22100 },
            { role: "user", sender: "en1gma", text: "ok thanks, will try that", ts: 30000 },
        ]
    },

    // Set 9: Network bridge setup (chunking comparison)
    chunking_1: {
        messages: [
            { role: "user", sender: "alcosta", text: "I want to setup bridged networking with Virtual Machines on ubuntu 20.04", ts: 0 },
            { role: "user", sender: "sarnold", text: "usual approach is to create a new bridge, add physical NICs, set IP on bridge", ts: 3000 },
            { role: "user", sender: "leftyfb", text: "alcosta: they should only need a network bridge", ts: 5000 },
            { role: "agent", sender: "helper", text: "alcosta: First run nmcli con add ifname br0 type bridge con-name br0", ts: 9000, chunkDelay: 2500 },
            { role: "agent", sender: "helper", text: "Then: nmcli con add type bridge-slave ifname enp6s0 master br0", ts: 11500, chunkDelay: 2500 },
            { role: "user", sender: "alcosta", text: "Something isn't right, the commands don't match my case", ts: 14000 },
            { role: "user", sender: "leftyfb", text: "it looks like you did lots of bad things, delete the bridge interfaces", ts: 16000 },
            { role: "agent", sender: "helper", text: "alcosta: Check nmcli con show to see current connections", ts: 20000, chunkDelay: 2500 },
            { role: "agent", sender: "helper", text: "Delete the broken ones with nmcli con delete <uuid>", ts: 22500, chunkDelay: 2500 },
            { role: "agent", sender: "helper", text: "Then start fresh with the bridge commands", ts: 25000, chunkDelay: 2500 },
            { role: "user", sender: "alcosta", text: "OK, deleted them. Now what?", ts: 28000 },
            { role: "user", sender: "leftyfb", text: "reboot, then br0 should have an IP address", ts: 30000 },
        ],
        fullChunked: true,
        comparisonSingleMessage: "alcosta: To set up bridged networking, run these commands in order: First, nmcli con add ifname br0 type bridge con-name br0. Then nmcli con add type bridge-slave ifname enp6s0 master br0. If you have existing broken bridge configs, delete them with nmcli con delete <uuid> first. You can check current connections with nmcli con show. After creating the bridge, reboot and br0 should get an IP address. Then configure your VMs to use br0 as the network interface."
    },

    // Set 10: GRUB/EFI boot repair (chunking comparison)
    chunking_2: {
        messages: [
            { role: "user", sender: "cahoots", text: "grub-install gives warning: EFI variables cannot be set on this system", ts: 0 },
            { role: "user", sender: "EriC^^", text: "in which mode are you booting? uefi? csm legacy?", ts: 3000 },
            { role: "user", sender: "cahoots", text: "I'm in uefi, csm legacy is disabled", ts: 5000 },
            { role: "agent", sender: "helper", text: "cahoots: First mount your root partition", ts: 9000, chunkDelay: 3000 },
            { role: "agent", sender: "helper", text: "sudo mount /dev/nvme0n1p3 /mnt", ts: 12000, chunkDelay: 3000 },
            { role: "user", sender: "EriC^^", text: "then run the for loop to bind mount /dev /proc /sys /run", ts: 14000 },
            { role: "agent", sender: "helper", text: "Next: for i in /dev /proc /sys /run; do sudo mount -R $i /mnt$i; done", ts: 18000, chunkDelay: 3000 },
            { role: "agent", sender: "helper", text: "Then: sudo chroot /mnt", ts: 21000, chunkDelay: 3000 },
            { role: "user", sender: "cahoots", text: "mount /boot/efi says can't find UUID", ts: 24000 },
            { role: "user", sender: "EriC^^", text: "the uuid changed, edit /etc/fstab with the new one from blkid", ts: 26000 },
            { role: "agent", sender: "helper", text: "cahoots: Run blkid to find the new UUID", ts: 30000, chunkDelay: 3000 },
            { role: "agent", sender: "helper", text: "Update /etc/fstab, then mount /boot/efi", ts: 33000, chunkDelay: 3000 },
            { role: "agent", sender: "helper", text: "Finally: grub-install --target=x86_64-efi", ts: 36000, chunkDelay: 3000 },
            { role: "user", sender: "cahoots", text: "installation finished, no error reported!", ts: 40000 },
        ],
        fullChunked: true,
        comparisonSingleMessage: "cahoots: To fix grub installation, follow these steps: Mount your root partition with sudo mount /dev/nvme0n1p3 /mnt. Then bind mount the virtual filesystems with for i in /dev /proc /sys /run; do sudo mount -R $i /mnt$i; done. Chroot into the system with sudo chroot /mnt. If mount /boot/efi fails due to UUID mismatch, run blkid to find the new UUID, update /etc/fstab with the correct UUID, then mount /boot/efi again. Finally run grub-install --target=x86_64-efi to install grub. After that, exit the chroot and reboot."
    }
};

// Export for use in app.js and generate_gifs.py
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { SURVEY_SETS, GIF_CONVERSATIONS };
}
