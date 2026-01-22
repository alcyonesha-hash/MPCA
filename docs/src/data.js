/**
 * Survey Data - 8 Comparison Sets
 *
 * Bilingual conversations (English base, Korean translation)
 *
 * Structure:
 * - Sets 1-2: Full vs NoTopicalFit (text)
 * - Sets 3-4: Full vs NoChunking (text)
 * - Sets 5-6: Full vs NoTiming (video)
 * - Sets 7-8: Full vs Baseline (text)
 *
 * Topics:
 * 1. Accounting system task assignment (NoTopicalFit)
 * 2. ChatGPT prompt engineering (NoTopicalFit)
 * 3. Filesystem check - IRC Ubuntu (Chunking)
 * 4. Gnome suspend issue - IRC Ubuntu (Chunking)
 * 5. Python version conflict (Timing - video)
 * 6. Excel VLOOKUP error (Timing - video)
 * 7. Book translation - literal vs liberal (Baseline)
 * 8. Database model selection (Baseline)
 */

const SURVEY_SETS = [
    // ============================================
    // Sets 1-2: Full vs NoTopicalFit
    // ============================================

    // Set 1: Accounting system task assignment
    {
        type: "text",
        comparisonType: "NoTopicalFit",
        topic: "Accounting system task assignment",
        topicKo: "회계 시스템 업무 배분",
        full: [
            { role: "user", sender: "jane", text: "How should we divide the new transaction system? Separate for payables and receivables?\n새 거래 시스템 담당 어떻게 나눌까요? 매입/매출 따로?", time: "09:15" },
            { role: "user", sender: "mike", text: "I've done payables before, I can take that\n저 매입 해봤으니까 맡을게요", time: "09:16" },
            { role: "user", sender: "sarah", text: "Then I'll handle receivables. By the way, I checked my payslip and overtime pay seems missing\n그럼 저는 매출 맡을게요. 근데 급여 명세서 봤는데 야근수당 누락된 것 같아요", time: "09:17" },
            { role: "agent", sender: "helper", text: "jane: Payables to Mike, receivables to Sarah works well\njane: 매입은 Mike, 매출은 Sarah로 배분하면 좋겠습니다", time: "09:18" },
            { role: "agent", sender: "helper", text: "You'll also need access permissions from IT\nIT팀에 권한 설정도 요청하세요", time: "09:18" },
            { role: "user", sender: "jane", text: "When do we need to finish the permission setup?\n권한 설정은 언제까지 해야 해요?", time: "09:19" },
            { role: "agent", sender: "helper", text: "jane: By Friday for Monday go-live\njane: 금요일까지 완료하면 월요일부터 운영 가능합니다", time: "09:20" },
            { role: "agent", sender: "helper", text: "sarah: For the missing overtime, contact HR with your timesheet records\nsarah: 야근수당 누락은 인사팀에 근무 기록과 함께 문의하세요", time: "09:21" },
        ],
        comparison: [
            { role: "user", sender: "jane", text: "How should we divide the new transaction system? Separate for payables and receivables?\n새 거래 시스템 담당 어떻게 나눌까요? 매입/매출 따로?", time: "09:15" },
            { role: "user", sender: "mike", text: "I've done payables before, I can take that\n저 매입 해봤으니까 맡을게요", time: "09:16" },
            { role: "user", sender: "sarah", text: "Then I'll handle receivables. By the way, I checked my payslip and overtime pay seems missing\n그럼 저는 매출 맡을게요. 근데 급여 명세서 봤는데 야근수당 누락된 것 같아요", time: "09:17" },
            { role: "agent", sender: "helper", text: "For missing overtime pay, submit your timesheet to HR\n야근수당 누락은 인사팀에 근무 기록을 제출하세요", time: "09:18" },
            { role: "agent", sender: "helper", text: "Payroll corrections are usually reflected next month\n급여 정정은 보통 다음 달에 반영됩니다", time: "09:18" },
            { role: "user", sender: "jane", text: "When do we need to finish the permission setup?\n권한 설정은 언제까지 해야 해요?", time: "09:19" },
            { role: "agent", sender: "helper", text: "For payroll inquiries, contact HR directly\n급여 문의는 인사팀에 직접 연락하세요", time: "09:20" },
        ]
    },

    // Set 2: ChatGPT prompt engineering
    {
        type: "text",
        comparisonType: "NoTopicalFit",
        topic: "ChatGPT prompt engineering",
        topicKo: "ChatGPT 프롬프트 작성",
        full: [
            { role: "user", sender: "tom", text: "How do I get better summaries from ChatGPT? Mine always come out too long\nChatGPT로 요약 잘 나오게 하려면 어떻게 해요? 맨날 너무 길어요", time: "14:30" },
            { role: "user", sender: "lisa", text: "Same here, I keep having to ask it to shorten\n저도요, 계속 줄여달라고 해야 해요", time: "14:31" },
            { role: "user", sender: "kevin", text: "I'm trying to integrate the API, what's the request format?\n저는 API 연동하려는데 요청 형식이 어떻게 돼요?", time: "14:32" },
            { role: "agent", sender: "helper", text: "tom: Specify length in your prompt like '3 sentences' or '100 words'\ntom: 프롬프트에 '3문장으로' 또는 '100단어로' 같이 길이를 명시하세요", time: "14:33" },
            { role: "agent", sender: "helper", text: "Role assignment helps too: 'You are a concise editor...'\n역할 부여도 효과적이에요: 'You are a concise editor...'", time: "14:33" },
            { role: "user", sender: "tom", text: "Can I also specify the output format?\n출력 형식도 지정할 수 있어요?", time: "14:34" },
            { role: "agent", sender: "helper", text: "tom: Yes, try 'respond in bullet points' or 'use markdown headers'\ntom: 네, 'bullet point로 작성해줘' 또는 'markdown 헤더 사용' 같이요", time: "14:35" },
            { role: "agent", sender: "helper", text: "kevin: POST to /v1/chat/completions with messages array in the body\nkevin: /v1/chat/completions에 messages 배열을 body로 POST 요청하세요", time: "14:36" },
        ],
        comparison: [
            { role: "user", sender: "tom", text: "How do I get better summaries from ChatGPT? Mine always come out too long\nChatGPT로 요약 잘 나오게 하려면 어떻게 해요? 맨날 너무 길어요", time: "14:30" },
            { role: "user", sender: "lisa", text: "Same here, I keep having to ask it to shorten\n저도요, 계속 줄여달라고 해야 해요", time: "14:31" },
            { role: "user", sender: "kevin", text: "I'm trying to integrate the API, what's the request format?\n저는 API 연동하려는데 요청 형식이 어떻게 돼요?", time: "14:32" },
            { role: "agent", sender: "helper", text: "API calls use POST to /v1/chat/completions endpoint\nAPI 호출은 /v1/chat/completions 엔드포인트에 POST 요청합니다", time: "14:33" },
            { role: "agent", sender: "helper", text: "Include Authorization header with Bearer token\nAuthorization 헤더에 Bearer 토큰을 포함하세요", time: "14:33" },
            { role: "user", sender: "tom", text: "Can I also specify the output format?\n출력 형식도 지정할 수 있어요?", time: "14:34" },
            { role: "agent", sender: "helper", text: "Extract result from response.choices[0].message.content\n응답에서 choices[0].message.content로 결과를 추출합니다", time: "14:35" },
        ]
    },

    // ============================================
    // Sets 3-4: Full vs NoChunking
    // ============================================

    // Set 3: Filesystem check - IRC Ubuntu based
    {
        type: "text",
        comparisonType: "NoChunking",
        topic: "Filesystem check after improper shutdown",
        topicKo: "비정상 종료 후 파일시스템 검사",
        full: [
            { role: "user", sender: "foo", text: "My old laptop keeps dropping to maintenance mode after improper shutdown. How can I force a filesystem check?\n노트북이 비정상 종료 후 유지보수 모드로 계속 떨어져요. 파일시스템 검사 어떻게 강제로 해요?", time: "20:24" },
            { role: "user", sender: "eric", text: "Which partition is your root?\n루트 파티션이 어디야?", time: "20:25" },
            { role: "user", sender: "foo", text: "sda2 I think\nsda2인 것 같아요", time: "20:26" },
            { role: "agent", sender: "helper", text: "foo: Run fsck /dev/sda2 to check the filesystem\nfoo: fsck /dev/sda2로 파일시스템 검사하세요", time: "20:27" },
            { role: "agent", sender: "helper", text: "Make sure the partition is unmounted first\n먼저 파티션이 마운트 해제되어 있어야 해요", time: "20:27" },
            { role: "user", sender: "foo", text: "It's asking to remove dirty bit, should I?\ndirty bit 제거하라는데 해도 되나요?", time: "20:28" },
            { role: "agent", sender: "helper", text: "foo: Yes, if fsck offers that option it means no corruption was found\nfoo: 네, fsck가 그 옵션을 제시하면 손상이 없다는 뜻이에요", time: "20:29" },
            { role: "agent", sender: "helper", text: "After that, check system logs for disk errors: journalctl -b -p err\n그 후 디스크 에러 로그 확인: journalctl -b -p err", time: "20:29" },
            { role: "agent", sender: "helper", text: "Also backup important data before further troubleshooting\n추가 작업 전에 중요 데이터 백업하세요", time: "20:30" },
        ],
        comparison: [
            { role: "user", sender: "foo", text: "My old laptop keeps dropping to maintenance mode after improper shutdown. How can I force a filesystem check?\n노트북이 비정상 종료 후 유지보수 모드로 계속 떨어져요. 파일시스템 검사 어떻게 강제로 해요?", time: "20:24" },
            { role: "user", sender: "eric", text: "Which partition is your root?\n루트 파티션이 어디야?", time: "20:25" },
            { role: "user", sender: "foo", text: "sda2 I think\nsda2인 것 같아요", time: "20:26" },
            { role: "agent", sender: "helper", text: "foo: Run fsck /dev/sda2 to check the filesystem. Make sure the partition is unmounted first. If it asks to remove the dirty bit, you can accept - this means no corruption was found. After the check, verify disk health with journalctl -b -p err to look for any hardware errors. You should also backup important data before any further troubleshooting since fsck repairs can sometimes be destructive.\nfoo: fsck /dev/sda2로 파일시스템 검사하세요. 먼저 파티션이 마운트 해제되어 있어야 합니다. dirty bit 제거를 묻는다면 승인해도 됩니다 - 손상이 없다는 뜻입니다. 검사 후 journalctl -b -p err로 하드웨어 에러 확인하세요. fsck 복구가 파괴적일 수 있으니 추가 작업 전 중요 데이터를 백업하세요.", time: "20:27" },
            { role: "user", sender: "foo", text: "It's asking to remove dirty bit, should I?\ndirty bit 제거하라는데 해도 되나요?", time: "20:28" },
            { role: "agent", sender: "helper", text: "Yes, as I mentioned, accepting the dirty bit removal is safe when fsck offers it.\n네, 말씀드렸듯이 fsck가 제시하면 dirty bit 제거는 안전합니다.", time: "20:29" },
        ]
    },

    // Set 4: Gnome suspend issue - IRC Ubuntu based
    {
        type: "text",
        comparisonType: "NoChunking",
        topic: "Gnome black screen after suspend",
        topicKo: "절전모드 후 검은 화면",
        full: [
            { role: "user", sender: "arkanoid", text: "Gnome shows black screen after resume from suspend. TTY and networking work fine\n절전모드 해제 후 Gnome이 검은 화면이에요. TTY랑 네트워크는 정상이에요", time: "14:58" },
            { role: "user", sender: "lotus", text: "Can you share your dmesg output?\ndmesg 출력 공유해줄 수 있어요?", time: "14:59" },
            { role: "user", sender: "arkanoid", text: "dmesg shows nothing suspicious. journalctl shows gnome-session GLib-GIO-CRITICAL error\ndmesg는 이상 없어요. journalctl에 gnome-session GLib-GIO-CRITICAL 에러가 있어요", time: "15:06" },
            { role: "agent", sender: "helper", text: "arkanoid: This looks like a known GDM/Gnome issue with suspend\narkanoid: 절전모드 관련 GDM/Gnome 알려진 이슈 같아요", time: "15:07" },
            { role: "agent", sender: "helper", text: "Try: sudo systemctl restart gdm3\n시도해보세요: sudo systemctl restart gdm3", time: "15:07" },
            { role: "user", sender: "arkanoid", text: "That brings login back but after login it goes black again\n로그인 화면은 나오는데 로그인하면 다시 검은 화면이에요", time: "15:10" },
            { role: "agent", sender: "helper", text: "arkanoid: Check if nvidia drivers are involved: lspci | grep -i nvidia\narkanoid: nvidia 드라이버 관련인지 확인: lspci | grep -i nvidia", time: "15:11" },
            { role: "agent", sender: "helper", text: "If nvidia, try: sudo prime-select intel, then reboot\nnvidia면 시도: sudo prime-select intel 후 재부팅", time: "15:11" },
            { role: "agent", sender: "helper", text: "The fix was backported in 22.10, consider upgrading if this persists\n수정 사항이 22.10에 적용됐으니 계속되면 업그레이드 고려하세요", time: "15:12" },
        ],
        comparison: [
            { role: "user", sender: "arkanoid", text: "Gnome shows black screen after resume from suspend. TTY and networking work fine\n절전모드 해제 후 Gnome이 검은 화면이에요. TTY랑 네트워크는 정상이에요", time: "14:58" },
            { role: "user", sender: "lotus", text: "Can you share your dmesg output?\ndmesg 출력 공유해줄 수 있어요?", time: "14:59" },
            { role: "user", sender: "arkanoid", text: "dmesg shows nothing suspicious. journalctl shows gnome-session GLib-GIO-CRITICAL error\ndmesg는 이상 없어요. journalctl에 gnome-session GLib-GIO-CRITICAL 에러가 있어요", time: "15:06" },
            { role: "agent", sender: "helper", text: "arkanoid: This is a known GDM/Gnome suspend issue. Try restarting GDM with 'sudo systemctl restart gdm3'. If that doesn't fully resolve it, check if NVIDIA drivers are involved using 'lspci | grep -i nvidia'. For NVIDIA systems, switching to Intel graphics with 'sudo prime-select intel' and rebooting often helps. This bug was fixed in Ubuntu 22.10, so upgrading is recommended if the issue persists.\narkanoid: GDM/Gnome 절전모드 관련 알려진 이슈입니다. 'sudo systemctl restart gdm3'로 GDM 재시작해보세요. 완전히 해결 안 되면 'lspci | grep -i nvidia'로 NVIDIA 드라이버 관련인지 확인하세요. NVIDIA 시스템은 'sudo prime-select intel'로 Intel 그래픽 전환 후 재부팅하면 도움됩니다. 이 버그는 22.10에서 수정됐으니 계속되면 업그레이드를 권장합니다.", time: "15:07" },
            { role: "user", sender: "arkanoid", text: "That brings login back but after login it goes black again\n로그인 화면은 나오는데 로그인하면 다시 검은 화면이에요", time: "15:10" },
            { role: "agent", sender: "helper", text: "As mentioned, try the NVIDIA workaround or consider upgrading to 22.10+.\n말씀드렸듯이 NVIDIA 해결책을 시도하거나 22.10+ 업그레이드를 고려하세요.", time: "15:11" },
        ]
    },

    // ============================================
    // Sets 5-6: Full vs NoTiming (Video)
    // ============================================

    // Set 5: Python version conflict
    {
        type: "video",
        comparisonType: "NoTiming",
        topic: "Python version library conflict",
        topicKo: "Python 버전별 라이브러리 충돌",
        full: {
            src: "data/gifs/timing_full_1.mp4",
            description: "Messages appear with natural delays"
        },
        comparison: {
            src: "data/gifs/timing_notiming_1.mp4",
            description: "Messages appear instantly"
        }
    },

    // Set 6: Excel VLOOKUP error
    {
        type: "video",
        comparisonType: "NoTiming",
        topic: "Excel VLOOKUP #N/A error",
        topicKo: "엑셀 VLOOKUP #N/A 에러",
        full: {
            src: "data/gifs/timing_full_2.mp4",
            description: "Messages appear with natural delays"
        },
        comparison: {
            src: "data/gifs/timing_notiming_2.mp4",
            description: "Messages appear instantly"
        }
    },

    // ============================================
    // Sets 7-8: Full vs Baseline
    // ============================================

    // Set 7: Book translation - literal vs liberal
    {
        type: "text",
        comparisonType: "Baseline",
        topic: "Book translation discussion",
        topicKo: "출판물 번역 의역/직역",
        full: [
            { role: "user", sender: "editor", text: "How should we translate 'I'm not crazy, my mother had me tested'? Direct translation sounds weird in Korean\n'I'm not crazy, my mother had me tested'를 어떻게 번역할까요? 직역하면 한국어로 이상해요", time: "11:20" },
            { role: "user", sender: "translatorA", text: "Literally it's '나 미친 거 아니야, 엄마가 검사받게 했어' - loses the humor\n직역하면 '나 미친 거 아니야, 엄마가 검사받게 했어' - 유머가 사라져요", time: "11:21" },
            { role: "user", sender: "translatorB", text: "Maybe '난 정상이야, 의사가 증명했어' captures the defensiveness better?\n'난 정상이야, 의사가 증명했어'가 방어적인 뉘앙스를 더 잘 살릴까요?", time: "11:22" },
            { role: "agent", sender: "helper", text: "editor: The humor comes from the absurd logic of using a test to prove sanity\neditor: 유머는 정신건강을 검사로 증명하려는 황당한 논리에서 와요", time: "11:23" },
            { role: "agent", sender: "helper", text: "Korean readers might not connect 'mother had me tested' with psychiatric evaluation\n한국 독자들은 '엄마가 검사받게 했어'를 정신과 검사와 연결 못 할 수 있어요", time: "11:23" },
            { role: "user", sender: "editor", text: "So should we add context or keep it vague?\n그럼 맥락을 추가할까요, 모호하게 둘까요?", time: "11:24" },
            { role: "agent", sender: "helper", text: "editor: Consider '난 미친 거 아냐, 병원에서 정상 판정 받았다고' - makes the joke clearer\neditor: '난 미친 거 아냐, 병원에서 정상 판정 받았다고' 고려해보세요 - 농담이 명확해져요", time: "11:25" },
            { role: "agent", sender: "helper", text: "The defensive 'my mother made me' becomes institutional '병원에서', which Korean readers understand as authority\n방어적인 'my mother made me'가 '병원에서'로 바뀌면 한국 독자들이 권위로 이해해요", time: "11:25" },
            { role: "agent", sender: "helper", text: "This preserves the character's need to prove themselves while being culturally accessible\n이렇게 하면 캐릭터의 증명 욕구를 살리면서 문화적으로 접근 가능해요", time: "11:26" },
        ],
        comparison: [
            { role: "user", sender: "editor", text: "How should we translate 'I'm not crazy, my mother had me tested'? Direct translation sounds weird in Korean\n'I'm not crazy, my mother had me tested'를 어떻게 번역할까요? 직역하면 한국어로 이상해요", time: "11:20" },
            { role: "user", sender: "translatorA", text: "Literally it's '나 미친 거 아니야, 엄마가 검사받게 했어' - loses the humor\n직역하면 '나 미친 거 아니야, 엄마가 검사받게 했어' - 유머가 사라져요", time: "11:21" },
            { role: "user", sender: "translatorB", text: "Maybe '난 정상이야, 의사가 증명했어' captures the defensiveness better?\n'난 정상이야, 의사가 증명했어'가 방어적인 뉘앙스를 더 잘 살릴까요?", time: "11:22" },
            { role: "agent", sender: "helper", text: "This phrase can be translated as '나는 미친 게 아니야, 엄마가 검사받게 했어' or adapted to '난 정상이야, 의사가 증명했어'. When translating idiomatic expressions, consider the target audience and cultural context. Literal translation preserves the original structure but may lose nuance. Liberal translation captures meaning but changes form. The best approach depends on whether the text prioritizes accuracy or readability. Consider the character's personality and the scene's tone when making this decision.\n이 문구는 '나는 미친 게 아니야, 엄마가 검사받게 했어' 또는 '난 정상이야, 의사가 증명했어'로 번역할 수 있습니다. 관용 표현을 번역할 때는 대상 독자와 문화적 맥락을 고려하세요. 직역은 원문 구조를 보존하지만 뉘앙스를 잃을 수 있습니다. 의역은 의미를 살리지만 형태가 바뀝니다. 최선의 접근법은 텍스트가 정확성과 가독성 중 무엇을 우선시하는지에 달려 있습니다.", time: "11:23" },
            { role: "user", sender: "editor", text: "So should we add context or keep it vague?\n그럼 맥락을 추가할까요, 모호하게 둘까요?", time: "11:24" },
            { role: "agent", sender: "helper", text: "Adding context depends on your translation philosophy and target readers.\n맥락 추가는 번역 철학과 대상 독자에 달려 있습니다.", time: "11:25" },
        ]
    },

    // Set 8: Database model selection
    {
        type: "text",
        comparisonType: "Baseline",
        topic: "Database model selection",
        topicKo: "외주 프로젝트 DB 모델 선택",
        full: [
            { role: "user", sender: "pm", text: "Reviewing the vendor's deliverable - why did they use MongoDB instead of PostgreSQL?\n외주 결과물 검토 중인데 왜 PostgreSQL 대신 MongoDB를 썼을까요?", time: "15:40" },
            { role: "user", sender: "dev", text: "Our data has clear relationships, wouldn't relational DB be better?\n우리 데이터는 관계가 명확한데 관계형 DB가 낫지 않나요?", time: "15:41" },
            { role: "agent", sender: "helper", text: "pm: MongoDB excels at flexible schemas and rapid prototyping\npm: MongoDB는 유연한 스키마와 빠른 프로토타이핑에 강해요", time: "15:42" },
            { role: "agent", sender: "helper", text: "If requirements were unclear early on, NoSQL makes iteration easier\n초기에 요구사항이 불명확했다면 NoSQL이 변경에 유리해요", time: "15:42" },
            { role: "user", sender: "pm", text: "But now we need complex joins for reporting. Is migration difficult?\n근데 이제 리포팅에 복잡한 조인이 필요해요. 마이그레이션 어려워요?", time: "15:43" },
            { role: "agent", sender: "helper", text: "pm: MongoDB can do lookups but they're slower than SQL joins\npm: MongoDB도 lookup 가능하지만 SQL join보다 느려요", time: "15:44" },
            { role: "agent", sender: "helper", text: "Migration complexity depends on how deeply nested your documents are\n마이그레이션 복잡도는 문서가 얼마나 중첩되어 있는지에 달려요", time: "15:44" },
            { role: "agent", sender: "helper", text: "Consider hybrid: keep MongoDB for raw data, use PostgreSQL for analytics\n하이브리드 고려: 원시 데이터는 MongoDB, 분석은 PostgreSQL로", time: "15:45" },
        ],
        comparison: [
            { role: "user", sender: "pm", text: "Reviewing the vendor's deliverable - why did they use MongoDB instead of PostgreSQL?\n외주 결과물 검토 중인데 왜 PostgreSQL 대신 MongoDB를 썼을까요?", time: "15:40" },
            { role: "user", sender: "dev", text: "Our data has clear relationships, wouldn't relational DB be better?\n우리 데이터는 관계가 명확한데 관계형 DB가 낫지 않나요?", time: "15:41" },
            { role: "agent", sender: "helper", text: "MongoDB and PostgreSQL serve different use cases. MongoDB is a NoSQL document database that stores data in JSON-like documents, offering flexible schemas. PostgreSQL is a relational database that uses tables with predefined schemas and supports complex queries with SQL. MongoDB is better for unstructured data and rapid development, while PostgreSQL excels at complex transactions and data integrity. The choice depends on your data structure, query patterns, and scalability requirements. Both databases have their strengths and are widely used in production environments.\nMongoDB와 PostgreSQL은 다른 용도로 사용됩니다. MongoDB는 JSON 형태 문서를 저장하는 NoSQL 문서 데이터베이스로 유연한 스키마를 제공합니다. PostgreSQL은 미리 정의된 스키마의 테이블을 사용하고 SQL로 복잡한 쿼리를 지원하는 관계형 데이터베이스입니다. MongoDB는 비정형 데이터와 빠른 개발에, PostgreSQL은 복잡한 트랜잭션과 데이터 무결성에 적합합니다.", time: "15:42" },
            { role: "user", sender: "pm", text: "But now we need complex joins for reporting. Is migration difficult?\n근데 이제 리포팅에 복잡한 조인이 필요해요. 마이그레이션 어려워요?", time: "15:43" },
            { role: "agent", sender: "helper", text: "Migration complexity varies depending on your specific data and requirements.\n마이그레이션 복잡도는 특정 데이터와 요구사항에 따라 다릅니다.", time: "15:44" },
        ]
    }
];

// GIF conversation data for generation script
const GIF_CONVERSATIONS = {
    // Set 5: Python version conflict (timing comparison)
    timing_1: {
        messages: [
            { role: "user", sender: "david", text: "Code that worked on Python 3.8 throws errors on 3.11\nPython 3.8에서 되던 코드가 3.11에서 에러 나요", ts: 0 },
            { role: "user", sender: "emma", text: "What kind of error?\n어떤 에러인데?", ts: 2500 },
            { role: "user", sender: "david", text: "AttributeError when importing tensorflow\ntensorflow import할 때 AttributeError요", ts: 5000 },
            { role: "agent", sender: "helper", text: "david: Your TensorFlow version might not support Python 3.11\ndavid: TensorFlow 버전이 Python 3.11을 지원 안 할 수 있어요", ts: 9000, noTimingTs: 5100 },
            { role: "agent", sender: "helper", text: "Check version with: pip show tensorflow\n버전 확인: pip show tensorflow", ts: 11500, noTimingTs: 5200 },
            { role: "user", sender: "david", text: "It says 2.10.0\n2.10.0이래요", ts: 14000 },
            { role: "agent", sender: "helper", text: "david: TF 2.10 only supports up to Python 3.10\ndavid: TF 2.10은 Python 3.10까지만 지원해요", ts: 18000, noTimingTs: 14100 },
            { role: "agent", sender: "helper", text: "Upgrade with: pip install tensorflow==2.15.0\n업그레이드: pip install tensorflow==2.15.0", ts: 21000, noTimingTs: 14200 },
            { role: "user", sender: "david", text: "Works now, thanks!\n됐어요, 감사합니다!", ts: 24000 },
        ]
    },

    // Set 6: Excel VLOOKUP error (timing comparison)
    timing_2: {
        messages: [
            { role: "user", sender: "amy", text: "My VLOOKUP keeps returning #N/A even though the value exists\nVLOOKUP이 값이 있는데도 #N/A가 나와요", ts: 0 },
            { role: "user", sender: "brian", text: "Are you using exact or approximate match?\n정확히 일치? 근사 일치?", ts: 2500 },
            { role: "user", sender: "amy", text: "I put TRUE at the end, isn't that exact match?\n끝에 TRUE 넣었는데 그게 정확히 일치 아니에요?", ts: 5000 },
            { role: "agent", sender: "helper", text: "amy: TRUE means approximate match, use FALSE for exact\namy: TRUE는 근사 일치예요, 정확히 일치는 FALSE 쓰세요", ts: 9000, noTimingTs: 5100 },
            { role: "agent", sender: "helper", text: "Approximate match also requires sorted data\n근사 일치는 데이터 정렬도 필요해요", ts: 11500, noTimingTs: 5200 },
            { role: "user", sender: "amy", text: "Changed to FALSE but still not working\nFALSE로 바꿨는데 여전히 안 돼요", ts: 14000 },
            { role: "agent", sender: "helper", text: "amy: There might be hidden spaces in your cells\namy: 셀에 숨겨진 공백이 있을 수 있어요", ts: 18000, noTimingTs: 14100 },
            { role: "agent", sender: "helper", text: "Try: =VLOOKUP(TRIM(A1),B:C,2,FALSE)\n시도: =VLOOKUP(TRIM(A1),B:C,2,FALSE)", ts: 21000, noTimingTs: 14200 },
            { role: "user", sender: "amy", text: "That was it! Hidden spaces were the problem\n그거였어요! 숨겨진 공백이 문제였네요", ts: 24000 },
        ]
    }
};

// Export for use in app.js and generate_gifs.py
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { SURVEY_SETS, GIF_CONVERSATIONS };
}
