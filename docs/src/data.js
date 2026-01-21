/**
 * Survey Data - 10 Comparison Sets
 *
 * Multi-party IRC-style conversations with interleaved threads
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

    // Set 1: Multiple users with package/dependency issues
    {
        type: "text",
        comparisonType: "Baseline",
        topic: "package installation help",
        full: [
            { role: "user", sender: "marcopolo", text: "apt-get says package not found for chromium", time: "14:23" },
            { role: "user", sender: "jenny_dev", text: "anyone know why pip install keeps failing?", time: "14:23" },
            { role: "user", sender: "n00b_linux", text: "is ubuntu 22.04 stable enough for daily use?", time: "14:23" },
            { role: "agent", sender: "helper", text: "marcopolo: Chromium is snap-only now.", time: "14:23 (+5s)" },
            { role: "agent", sender: "helper", text: "Use: snap install chromium", time: "14:23 (+2s)" },
            { role: "user", sender: "marcopolo", text: "oh that explains it thanks", time: "14:24" },
            { role: "agent", sender: "helper", text: "jenny_dev: What error are you getting?", time: "14:24 (+4s)" },
            { role: "user", sender: "jenny_dev", text: "permission denied on /usr/lib", time: "14:24" },
            { role: "user", sender: "techguru42", text: "22.04 is solid, been using it for months", time: "14:25" },
            { role: "agent", sender: "helper", text: "jenny_dev: Use pip install --user or venv", time: "14:25 (+3s)" },
        ],
        comparison: [
            { role: "user", sender: "marcopolo", text: "apt-get says package not found for chromium", time: "14:23" },
            { role: "user", sender: "jenny_dev", text: "anyone know why pip install keeps failing?", time: "14:23" },
            { role: "user", sender: "n00b_linux", text: "is ubuntu 22.04 stable enough for daily use?", time: "14:23" },
            { role: "agent", sender: "helper", text: "For the Chromium issue, you should know that Chromium browser is no longer available via apt on Ubuntu. It's now distributed exclusively as a snap package. You can install it using 'sudo snap install chromium'. For the pip issue, when you see permission denied errors, it usually means you're trying to install packages system-wide without proper permissions. You have several options: use 'pip install --user packagename' to install to your home directory, create a virtual environment with 'python -m venv myenv' and activate it, or use pipx for CLI tools. As for Ubuntu 22.04, it's an LTS release and quite stable for daily use.", time: "14:23 (+0s)" },
            { role: "user", sender: "marcopolo", text: "oh that explains it thanks", time: "14:24" },
            { role: "user", sender: "jenny_dev", text: "permission denied on /usr/lib", time: "14:24" },
            { role: "user", sender: "techguru42", text: "22.04 is solid, been using it for months", time: "14:25" },
            { role: "agent", sender: "helper", text: "Yes, as I mentioned, use pip with --user flag or set up a virtual environment. Virtual environments are the recommended approach for Python development as they isolate project dependencies.", time: "14:25 (+0s)" },
        ]
    },

    // Set 2: Server troubleshooting with multiple threads
    {
        type: "text",
        comparisonType: "Baseline",
        topic: "server issues",
        full: [
            { role: "user", sender: "sysadmin_joe", text: "nginx keeps returning 502, any ideas?", time: "09:15" },
            { role: "user", sender: "devops_kim", text: "how do I check if port 8080 is in use?", time: "09:15" },
            { role: "user", sender: "webmaster", text: "my SSL cert expired, how to renew with certbot?", time: "09:16" },
            { role: "agent", sender: "helper", text: "sysadmin_joe: Check upstream service status", time: "09:16 (+4s)" },
            { role: "agent", sender: "helper", text: "Is your backend actually running?", time: "09:16 (+2s)" },
            { role: "user", sender: "sysadmin_joe", text: "oh the php-fpm crashed", time: "09:17" },
            { role: "agent", sender: "helper", text: "devops_kim: lsof -i :8080", time: "09:17 (+3s)" },
            { role: "agent", sender: "helper", text: "or: ss -tlnp | grep 8080", time: "09:17 (+2s)" },
            { role: "user", sender: "devops_kim", text: "perfect, found it", time: "09:18" },
            { role: "agent", sender: "helper", text: "webmaster: certbot renew --dry-run first", time: "09:18 (+5s)" },
        ],
        comparison: [
            { role: "user", sender: "sysadmin_joe", text: "nginx keeps returning 502, any ideas?", time: "09:15" },
            { role: "user", sender: "devops_kim", text: "how do I check if port 8080 is in use?", time: "09:15" },
            { role: "user", sender: "webmaster", text: "my SSL cert expired, how to renew with certbot?", time: "09:16" },
            { role: "agent", sender: "helper", text: "For the 502 error, this typically means nginx can't connect to your upstream server (like PHP-FPM, Node.js, etc.). Check if your backend service is running with 'systemctl status php-fpm' or whatever service you're using. Also check nginx error logs at /var/log/nginx/error.log. For checking port usage, you can use 'lsof -i :8080' or 'ss -tlnp | grep 8080' or 'netstat -tlnp | grep 8080'. For SSL renewal with certbot, run 'certbot renew --dry-run' first to test, then 'certbot renew' if successful. You might need to reload nginx after.", time: "09:16 (+0s)" },
            { role: "user", sender: "sysadmin_joe", text: "oh the php-fpm crashed", time: "09:17" },
            { role: "user", sender: "devops_kim", text: "perfect, found it", time: "09:18" },
            { role: "agent", sender: "helper", text: "Glad that helped! For future reference, you can set up monitoring for PHP-FPM to catch crashes early. Tools like monit or systemd's restart functionality can automatically restart services when they fail.", time: "09:18 (+0s)" },
        ]
    },

    // Set 3: Git and development workflow
    {
        type: "text",
        comparisonType: "Baseline",
        topic: "git workflow",
        full: [
            { role: "user", sender: "newbie_coder", text: "I accidentally committed to main, how to undo?", time: "16:42" },
            { role: "user", sender: "alice_dev", text: "whats the diff between rebase and merge?", time: "16:42" },
            { role: "user", sender: "bob_senior", text: "use rebase for cleaner history imo", time: "16:43" },
            { role: "agent", sender: "helper", text: "newbie_coder: git reset HEAD~1", time: "16:43 (+4s)" },
            { role: "agent", sender: "helper", text: "This keeps your changes unstaged", time: "16:43 (+2s)" },
            { role: "user", sender: "newbie_coder", text: "what if I already pushed?", time: "16:44" },
            { role: "agent", sender: "helper", text: "Then you need force push, but be careful", time: "16:44 (+5s)" },
            { role: "user", sender: "alice_dev", text: "so rebase rewrites history?", time: "16:45" },
            { role: "agent", sender: "helper", text: "alice_dev: Yes, rebase replays commits", time: "16:45 (+3s)" },
            { role: "agent", sender: "helper", text: "Merge preserves original history", time: "16:45 (+2s)" },
        ],
        comparison: [
            { role: "user", sender: "newbie_coder", text: "I accidentally committed to main, how to undo?", time: "16:42" },
            { role: "user", sender: "alice_dev", text: "whats the diff between rebase and merge?", time: "16:42" },
            { role: "user", sender: "bob_senior", text: "use rebase for cleaner history imo", time: "16:43" },
            { role: "agent", sender: "helper", text: "To undo a commit on main, use 'git reset HEAD~1' which will undo the last commit but keep your changes. If you want to completely discard the changes, use 'git reset --hard HEAD~1'. If you already pushed, you'll need 'git push --force' but be very careful as this rewrites remote history. For rebase vs merge: rebase rewrites commit history by replaying your commits on top of another branch, creating a linear history. Merge creates a merge commit that combines two branches, preserving the original branching structure. Rebase is cleaner but changes commit hashes; merge is safer for shared branches.", time: "16:43 (+0s)" },
            { role: "user", sender: "newbie_coder", text: "what if I already pushed?", time: "16:44" },
            { role: "user", sender: "alice_dev", text: "so rebase rewrites history?", time: "16:45" },
            { role: "agent", sender: "helper", text: "Yes, as I explained, if you pushed you need force push. And yes, rebase does rewrite history which is why you should never rebase commits that others might have pulled.", time: "16:45 (+0s)" },
        ]
    },

    // Set 4: Docker and containerization
    {
        type: "text",
        comparisonType: "Baseline",
        topic: "docker help",
        full: [
            { role: "user", sender: "docker_noob", text: "my container keeps exiting immediately", time: "11:30" },
            { role: "user", sender: "k8s_fan", text: "whats the best way to pass env vars to docker?", time: "11:30" },
            { role: "user", sender: "fullstack_dev", text: "how do I connect containers on same network?", time: "11:31" },
            { role: "agent", sender: "helper", text: "docker_noob: Check logs with docker logs <id>", time: "11:31 (+5s)" },
            { role: "user", sender: "docker_noob", text: "says no main process", time: "11:32" },
            { role: "agent", sender: "helper", text: "Your CMD needs a foreground process", time: "11:32 (+4s)" },
            { role: "agent", sender: "helper", text: "k8s_fan: Use -e flag or --env-file", time: "11:32 (+2s)" },
            { role: "user", sender: "k8s_fan", text: "can I use .env file directly?", time: "11:33" },
            { role: "agent", sender: "helper", text: "Yes: docker run --env-file .env", time: "11:33 (+3s)" },
            { role: "agent", sender: "helper", text: "fullstack_dev: Create a network first", time: "11:34 (+4s)" },
        ],
        comparison: [
            { role: "user", sender: "docker_noob", text: "my container keeps exiting immediately", time: "11:30" },
            { role: "user", sender: "k8s_fan", text: "whats the best way to pass env vars to docker?", time: "11:30" },
            { role: "user", sender: "fullstack_dev", text: "how do I connect containers on same network?", time: "11:31" },
            { role: "agent", sender: "helper", text: "When a container exits immediately, it usually means the main process finished or crashed. Check logs with 'docker logs <container_id>'. Common causes include: no foreground process (the CMD runs and exits), crash on startup, or missing dependencies. For environment variables, you can use '-e VAR=value' for individual vars, '--env-file .env' to load from a file, or define them in docker-compose.yml. For container networking, create a custom network with 'docker network create mynet' then run containers with '--network mynet' and they can reach each other by container name.", time: "11:31 (+0s)" },
            { role: "user", sender: "docker_noob", text: "says no main process", time: "11:32" },
            { role: "user", sender: "k8s_fan", text: "can I use .env file directly?", time: "11:33" },
            { role: "agent", sender: "helper", text: "As I mentioned, you need a foreground process. Make sure your CMD runs something that doesn't exit, like a server. And yes, you can use .env files directly with the --env-file flag I mentioned earlier.", time: "11:33 (+0s)" },
        ]
    },

    // ============================================
    // Sets 5-6: Full vs No Timing (GIF)
    // ============================================

    // Set 5: Multi-user quick help
    {
        type: "gif",
        comparisonType: "NoTiming",
        topic: "multi-user quick help",
        full: {
            src: "data/gifs/timing_full_1.gif",
            description: "Messages appear with natural delays"
        },
        comparison: {
            src: "data/gifs/timing_notiming_1.gif",
            description: "Messages appear instantly"
        }
    },

    // Set 6: Interleaved technical questions
    {
        type: "gif",
        comparisonType: "NoTiming",
        topic: "interleaved questions",
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

    // Set 7: Multiple threads, agent tracks correctly vs confused
    {
        type: "text",
        comparisonType: "NoTopicalFit",
        topic: "multiple support threads",
        full: [
            { role: "user", sender: "mike_ops", text: "redis keeps running out of memory", time: "15:01" },
            { role: "user", sender: "sarah_fe", text: "how do I center a div vertically?", time: "15:01" },
            { role: "user", sender: "tom_backend", text: "postgres query is super slow", time: "15:02" },
            { role: "agent", sender: "helper", text: "mike_ops: Check maxmemory setting", time: "15:02 (+4s)" },
            { role: "agent", sender: "helper", text: "Also set maxmemory-policy", time: "15:02 (+2s)" },
            { role: "user", sender: "sarah_fe", text: "using flexbox btw", time: "15:03" },
            { role: "agent", sender: "helper", text: "sarah_fe: align-items: center on parent", time: "15:03 (+3s)" },
            { role: "user", sender: "mike_ops", text: "its set to 2gb, should I increase?", time: "15:04" },
            { role: "agent", sender: "helper", text: "mike_ops: Depends on your data size", time: "15:04 (+4s)" },
            { role: "agent", sender: "helper", text: "tom_backend: Run EXPLAIN ANALYZE on query", time: "15:05 (+3s)" },
        ],
        comparison: [
            { role: "user", sender: "mike_ops", text: "redis keeps running out of memory", time: "15:01" },
            { role: "user", sender: "sarah_fe", text: "how do I center a div vertically?", time: "15:01" },
            { role: "user", sender: "tom_backend", text: "postgres query is super slow", time: "15:02" },
            { role: "agent", sender: "helper", text: "For centering, use flexbox with align-items: center", time: "15:02 (+0s)" },
            { role: "user", sender: "sarah_fe", text: "using flexbox btw", time: "15:03" },
            { role: "agent", sender: "helper", text: "Great, flexbox is the best approach!", time: "15:03 (+0s)" },
            { role: "user", sender: "mike_ops", text: "its set to 2gb, should I increase?", time: "15:04" },
            { role: "agent", sender: "helper", text: "2gb should be enough for most flexbox layouts", time: "15:04 (+0s)" },
            { role: "agent", sender: "helper", text: "CSS doesn't usually need that much memory", time: "15:05 (+0s)" },
        ]
    },

    // Set 8: Complex multi-thread conversation
    {
        type: "text",
        comparisonType: "NoTopicalFit",
        topic: "mixed development help",
        full: [
            { role: "user", sender: "junior_dev", text: "why does my React component keep re-rendering?", time: "10:20" },
            { role: "user", sender: "data_guy", text: "best way to handle CSV with 1M rows in python?", time: "10:20" },
            { role: "user", sender: "mobile_dev", text: "flutter or react native in 2024?", time: "10:21" },
            { role: "agent", sender: "helper", text: "junior_dev: Check your useEffect deps", time: "10:21 (+5s)" },
            { role: "user", sender: "junior_dev", text: "its an empty array", time: "10:22" },
            { role: "agent", sender: "helper", text: "Then check parent re-renders", time: "10:22 (+3s)" },
            { role: "agent", sender: "helper", text: "data_guy: Use pandas with chunks", time: "10:22 (+2s)" },
            { role: "user", sender: "data_guy", text: "pd.read_csv with chunksize?", time: "10:23" },
            { role: "agent", sender: "helper", text: "data_guy: Exactly, or use dask", time: "10:23 (+4s)" },
            { role: "user", sender: "senior_eng", text: "flutter has better perf imo", time: "10:24" },
        ],
        comparison: [
            { role: "user", sender: "junior_dev", text: "why does my React component keep re-rendering?", time: "10:20" },
            { role: "user", sender: "data_guy", text: "best way to handle CSV with 1M rows in python?", time: "10:20" },
            { role: "user", sender: "mobile_dev", text: "flutter or react native in 2024?", time: "10:21" },
            { role: "agent", sender: "helper", text: "Flutter has really improved recently!", time: "10:21 (+0s)" },
            { role: "user", sender: "junior_dev", text: "its an empty array", time: "10:22" },
            { role: "agent", sender: "helper", text: "Empty arrays are common in Flutter too", time: "10:22 (+0s)" },
            { role: "user", sender: "data_guy", text: "pd.read_csv with chunksize?", time: "10:23" },
            { role: "agent", sender: "helper", text: "React Native also supports chunked loading", time: "10:23 (+0s)" },
            { role: "user", sender: "senior_eng", text: "flutter has better perf imo", time: "10:24" },
            { role: "agent", sender: "helper", text: "Yes Flutter performance is great for mobile!", time: "10:24 (+0s)" },
        ]
    },

    // ============================================
    // Sets 9-10: Full vs No Chunking (GIF)
    // ============================================

    // Set 9: Multi-user with chunked responses
    {
        type: "gif",
        comparisonType: "NoChunking",
        topic: "server setup help",
        full: {
            src: "data/gifs/chunking_full_1.gif",
            description: "Response split into natural chunks"
        },
        comparison: {
            src: "data/gifs/chunking_nochunk_1.gif",
            description: "Single long message"
        }
    },

    // Set 10: Detailed explanation chunked vs wall of text
    {
        type: "gif",
        comparisonType: "NoChunking",
        topic: "deployment steps",
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
