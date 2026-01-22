/**
 * Survey Data - 8 Comparison Sets
 *
 * Bilingual conversations (English base, Korean translation)
 *
 * Structure:
 * - Sets 1-2: Full vs NoTopicalFit (text)
 * - Sets 3-4: Full vs NoChunking (text)
 * - Sets 5-6: Full vs NoTiming (video)
 * - Sets 7-8: Full vs Baseline (video)
 *
 * Topics:
 * 1. Procurement team invoice/shipment task (NoTopicalFit)
 * 2. GPT hallucination + API error (NoTopicalFit)
 * 3. Filesystem check - IRC Ubuntu (Chunking)
 * 4. Gnome suspend issue - IRC Ubuntu (Chunking)
 * 5. Python TensorFlow version conflict (Timing - video)
 * 6. Excel VLOOKUP error (Timing - video)
 * 7. Book translation - Friends quote (Baseline - video)
 * 8. Database model selection (Baseline - video)
 */

const SURVEY_SETS = [
    // ============================================
    // Sets 1-2: Full vs NoTopicalFit
    // ============================================

    // Set 1: Procurement team invoice/shipment task assignment
    {
        type: "text",
        comparisonType: "NoTopicalFit",
        topic: "Procurement invoice and shipment tasks",
        topicKo: "구매팀 송장/출장 업무 배분",
        full: [
            { role: "user", sender: "jane", text: "We need to assign the new vendor invoice verification and shipment inspection trip. Any volunteers?\n새 거래처 송장 확인이랑 선적지 출장 업무 배분해야 해요. 누가 할래요?", time: "09:15" },
            { role: "user", sender: "mike", text: "I can take the invoice verification, I handled similar cases before\n송장 확인은 제가 할게요, 비슷한 거 해봤어요", time: "09:16" },
            { role: "user", sender: "sarah", text: "I'll handle the shipment inspection then\n그럼 선적지 출장은 제가 갈게요", time: "09:17" },
            { role: "user", sender: "kevin", text: "Hey everyone, I just checked my payslip and last month's overtime seems to be missing\n다들 안녕, 방금 급여 명세서 봤는데 저번 달 야근수당이 누락된 것 같아요", time: "09:18" },
            { role: "agent", sender: "helper", text: "jane: Mike on invoices, Sarah on shipment inspection works well\njane: 송장은 Mike, 선적 출장은 Sarah로 배분하면 좋겠습니다", time: "09:19" },
            { role: "agent", sender: "helper", text: "Sarah will need travel approval from admin by Thursday\nSarah는 목요일까지 관리팀에 출장 승인 받아야 해요", time: "09:19" },
            { role: "user", sender: "jane", text: "What documents does Mike need from the vendor?\nMike가 거래처에서 받아야 할 서류가 뭐예요?", time: "09:20" },
            { role: "agent", sender: "helper", text: "jane: Commercial invoice, packing list, and certificate of origin\njane: 상업송장, 패킹리스트, 원산지증명서가 필요해요", time: "09:21" },
        ],
        comparison: [
            { role: "user", sender: "jane", text: "We need to assign the new vendor invoice verification and shipment inspection trip. Any volunteers?\n새 거래처 송장 확인이랑 선적지 출장 업무 배분해야 해요. 누가 할래요?", time: "09:15" },
            { role: "user", sender: "mike", text: "I can take the invoice verification, I handled similar cases before\n송장 확인은 제가 할게요, 비슷한 거 해봤어요", time: "09:16" },
            { role: "user", sender: "sarah", text: "I'll handle the shipment inspection then\n그럼 선적지 출장은 제가 갈게요", time: "09:17" },
            { role: "user", sender: "kevin", text: "Hey everyone, I just checked my payslip and last month's overtime seems to be missing\n다들 안녕, 방금 급여 명세서 봤는데 저번 달 야근수당이 누락된 것 같아요", time: "09:18" },
            { role: "agent", sender: "helper", text: "kevin: For missing overtime, submit your timesheet to HR\nkevin: 야근수당 누락은 인사팀에 근무 기록을 제출하세요", time: "09:19" },
            { role: "agent", sender: "helper", text: "Payroll corrections are usually reflected next month\n급여 정정은 보통 다음 달에 반영됩니다", time: "09:19" },
            { role: "user", sender: "jane", text: "What documents does Mike need from the vendor?\nMike가 거래처에서 받아야 할 서류가 뭐예요?", time: "09:20" },
            { role: "agent", sender: "helper", text: "kevin: Contact HR directly for payroll inquiries\nkevin: 급여 문의는 인사팀에 직접 연락하세요", time: "09:21" },
        ]
    },

    // Set 2: GPT hallucination + API error
    {
        type: "text",
        comparisonType: "NoTopicalFit",
        topic: "GPT reference hallucination and API",
        topicKo: "GPT 레퍼런스 할루시네이션 + API",
        full: [
            { role: "user", sender: "tom", text: "I asked GPT for academic references and it gave me papers that don't exist. How do I prevent this?\nGPT한테 학술 레퍼런스 달라고 했더니 없는 논문을 줬어요. 이거 어떻게 방지해요?", time: "14:30" },
            { role: "user", sender: "lisa", text: "Same happened to me, it even made up author names\n저도요, 저자 이름도 지어냈어요", time: "14:31" },
            { role: "user", sender: "kevin", text: "I'm getting 429 errors when calling the API, what's the rate limit?\n저는 API 호출할 때 429 에러가 나는데 rate limit이 어떻게 되나요?", time: "14:32" },
            { role: "agent", sender: "helper", text: "tom: GPT doesn't have real-time database access, it generates plausible-sounding references\ntom: GPT는 실시간 DB 접근이 없어서 그럴듯한 레퍼런스를 생성해요", time: "14:33" },
            { role: "agent", sender: "helper", text: "Always verify citations with Google Scholar or the actual journal\n항상 Google Scholar나 실제 저널에서 인용 확인하세요", time: "14:33" },
            { role: "user", sender: "tom", text: "Is there any way to make it more reliable for citations?\n인용을 더 신뢰할 수 있게 하는 방법 없어요?", time: "14:34" },
            { role: "agent", sender: "helper", text: "tom: Use RAG (Retrieval-Augmented Generation) or provide real papers in the prompt\ntom: RAG 사용하거나 실제 논문을 프롬프트에 제공하세요", time: "14:35" },
            { role: "agent", sender: "helper", text: "kevin: Rate limit is 3 RPM for free tier, 60 RPM for pay-as-you-go\nkevin: Rate limit은 무료는 분당 3회, 유료는 분당 60회예요", time: "14:36" },
        ],
        comparison: [
            { role: "user", sender: "tom", text: "I asked GPT for academic references and it gave me papers that don't exist. How do I prevent this?\nGPT한테 학술 레퍼런스 달라고 했더니 없는 논문을 줬어요. 이거 어떻게 방지해요?", time: "14:30" },
            { role: "user", sender: "lisa", text: "Same happened to me, it even made up author names\n저도요, 저자 이름도 지어냈어요", time: "14:31" },
            { role: "user", sender: "kevin", text: "I'm getting 429 errors when calling the API, what's the rate limit?\n저는 API 호출할 때 429 에러가 나는데 rate limit이 어떻게 되나요?", time: "14:32" },
            { role: "agent", sender: "helper", text: "kevin: 429 error means you've hit the rate limit\nkevin: 429 에러는 rate limit에 도달했다는 뜻이에요", time: "14:33" },
            { role: "agent", sender: "helper", text: "Add exponential backoff to your API calls\nAPI 호출에 exponential backoff를 추가하세요", time: "14:33" },
            { role: "user", sender: "tom", text: "Is there any way to make it more reliable for citations?\n인용을 더 신뢰할 수 있게 하는 방법 없어요?", time: "14:34" },
            { role: "agent", sender: "helper", text: "kevin: Check your usage dashboard to monitor API quota\nkevin: 사용량 대시보드에서 API 할당량 확인하세요", time: "14:35" },
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

    // Set 5: Python TensorFlow version conflict
    {
        type: "video",
        comparisonType: "NoTiming",
        topic: "Python TensorFlow version conflict",
        topicKo: "Python TensorFlow 버전 충돌",
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
    // Sets 7-8: Full vs Baseline (Video)
    // ============================================

    // Set 7: Book translation - Friends quote
    {
        type: "video",
        comparisonType: "Baseline",
        topic: "Book translation discussion",
        topicKo: "출판물 번역 논의",
        full: {
            src: "data/gifs/baseline_full_1.mp4",
            description: "Agent with contextual responses"
        },
        comparison: {
            src: "data/gifs/baseline_baseline_1.mp4",
            description: "Agent with generic responses"
        }
    },

    // Set 8: Database model selection
    {
        type: "video",
        comparisonType: "Baseline",
        topic: "Database model selection",
        topicKo: "DB 모델 선택",
        full: {
            src: "data/gifs/baseline_full_2.mp4",
            description: "Agent with contextual responses"
        },
        comparison: {
            src: "data/gifs/baseline_baseline_2.mp4",
            description: "Agent with generic responses"
        }
    }
];

// GIF conversation data for generation script
const GIF_CONVERSATIONS = {
    // Set 5: Python TensorFlow version conflict (timing comparison)
    timing_1: {
        messages: [
            { role: "user", sender: "david", text: "My TensorFlow code that worked on Python 3.8 throws AttributeError on 3.11\nPython 3.8에서 되던 TensorFlow 코드가 3.11에서 AttributeError 나요", ts: 0 },
            { role: "user", sender: "emma", text: "What error exactly?\n정확히 무슨 에러?", ts: 2500 },
            { role: "user", sender: "david", text: "AttributeError: module 'tensorflow' has no attribute 'Session'\nAttributeError: module 'tensorflow' has no attribute 'Session'이요", ts: 5000 },
            { role: "agent", sender: "helper", text: "david: tf.Session was removed in TF2. Use tf.compat.v1.Session instead\ndavid: tf.Session은 TF2에서 삭제됐어요. tf.compat.v1.Session 쓰세요", ts: 9000, noTimingTs: 5100 },
            { role: "agent", sender: "helper", text: "Or migrate to TF2 eager execution\n아니면 TF2 즉시 실행으로 마이그레이션하세요", ts: 11500, noTimingTs: 5200 },
            { role: "user", sender: "david", text: "I'm using TensorFlow 2.10. Is that the issue?\nTensorFlow 2.10 쓰는데 그게 문제인가요?", ts: 14000 },
            { role: "agent", sender: "helper", text: "david: TF 2.10 only supports up to Python 3.10\ndavid: TF 2.10은 Python 3.10까지만 지원해요", ts: 18000, noTimingTs: 14100 },
            { role: "agent", sender: "helper", text: "Upgrade to TF 2.15: pip install tensorflow==2.15.0\nTF 2.15로 업그레이드: pip install tensorflow==2.15.0", ts: 21000, noTimingTs: 14200 },
            { role: "user", sender: "david", text: "Works now with 2.15, thanks!\n2.15로 하니까 돼요, 감사합니다!", ts: 24000 },
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
    },

    // Set 7: Book translation - Friends "We were on a break!" (baseline comparison)
    baseline_1: {
        messages: [
            { role: "user", sender: "editor", text: "How should we translate 'We were on a break!' from Friends? It's a running joke\n프렌즈의 'We were on a break!' 어떻게 번역할까요? 반복되는 유머예요", ts: 0 },
            { role: "user", sender: "translatorA", text: "Literally '우리 그때 잠깐 헤어졌었잖아!' - but it doesn't hit the same\n직역하면 '우리 그때 잠깐 헤어졌었잖아!' - 근데 임팩트가 달라요", ts: 2500 },
            { role: "user", sender: "translatorB", text: "'잠시 쉬는 거였다고!' sounds more defensive like Ross\n'잠시 쉬는 거였다고!'가 Ross처럼 방어적으로 들려요", ts: 5000 },
        ],
        fullMessages: [
            { role: "agent", sender: "helper", text: "editor: The humor relies on Ross repeating this defensively for years\neditor: 유머는 Ross가 수년간 방어적으로 반복하는 데서 와요", ts: 9000, noTimingTs: 5100 },
            { role: "agent", sender: "helper", text: "Korean viewers need to feel his desperation, not just understand the words\n한국 시청자들이 단어가 아닌 그의 절박함을 느껴야 해요", ts: 11500, noTimingTs: 5200 },
            { role: "user", sender: "editor", text: "So which captures that desperation better?\n그럼 그 절박함을 더 잘 살리는 건 뭐예요?", ts: 14000 },
            { role: "agent", sender: "helper", text: "editor: '잠깐 쉬는 거였다고!' with the emphasis marker keeps the exasperation\neditor: '잠깐 쉬는 거였다고!'에 강조 어미를 쓰면 짜증이 살아요", ts: 18000, noTimingTs: 14100 },
            { role: "agent", sender: "helper", text: "The '다고' ending in Korean conveys 'I keep telling you this!'\n한국어의 '다고' 어미가 '계속 말하잖아!'를 전달해요", ts: 21000, noTimingTs: 14200 },
            { role: "agent", sender: "helper", text: "Consistency matters too - use the same phrase every time Ross says it\n일관성도 중요해요 - Ross가 말할 때마다 같은 표현을 쓰세요", ts: 24000, noTimingTs: 14300 },
        ],
        baselineMessages: [
            { role: "agent", sender: "helper", text: "This phrase can be translated as '우리 그때 헤어진 거였어!' or '잠시 쉬는 거였다고!'. When translating idiomatic expressions, consider the target audience and cultural context. Literal translation preserves the original structure but may lose nuance. Liberal translation captures meaning but changes form. The best approach depends on whether the text prioritizes accuracy or readability.\n이 문구는 '우리 그때 헤어진 거였어!' 또는 '잠시 쉬는 거였다고!'로 번역할 수 있습니다. 관용 표현을 번역할 때는 대상 독자와 문화적 맥락을 고려하세요. 직역은 원문 구조를 보존하지만 뉘앙스를 잃을 수 있습니다. 의역은 의미를 살리지만 형태가 바뀝니다.", ts: 9000, noTimingTs: 5100 },
            { role: "user", sender: "editor", text: "So which captures that desperation better?\n그럼 그 절박함을 더 잘 살리는 건 뭐예요?", ts: 14000 },
            { role: "agent", sender: "helper", text: "Translation choice depends on your translation philosophy and target readers.\n번역 선택은 번역 철학과 대상 독자에 달려 있습니다.", ts: 18000, noTimingTs: 14100 },
        ]
    },

    // Set 8: Database model selection - Big Bang Theory "Bazinga!" context (baseline comparison)
    baseline_2: {
        messages: [
            { role: "user", sender: "pm", text: "The vendor used MongoDB but we need complex joins for reporting. Should we migrate to PostgreSQL?\n외주가 MongoDB 썼는데 리포팅에 복잡한 조인 필요해요. PostgreSQL로 마이그레이션해야 하나요?", ts: 0 },
            { role: "user", sender: "dev", text: "Our e-commerce data has clear product-order-customer relationships\n우리 이커머스 데이터는 상품-주문-고객 관계가 명확해요", ts: 2500 },
            { role: "user", sender: "analyst", text: "I need to run monthly sales reports with multiple table joins\n저는 여러 테이블 조인으로 월간 매출 리포트 돌려야 해요", ts: 5000 },
        ],
        fullMessages: [
            { role: "agent", sender: "helper", text: "pm: MongoDB lookups work but are 3-5x slower than SQL joins for your use case\npm: MongoDB lookup도 되지만 이 경우 SQL join보다 3-5배 느려요", ts: 9000, noTimingTs: 5100 },
            { role: "agent", sender: "helper", text: "For e-commerce with clear relationships, PostgreSQL is the better fit\n명확한 관계가 있는 이커머스엔 PostgreSQL이 더 맞아요", ts: 11500, noTimingTs: 5200 },
            { role: "user", sender: "pm", text: "How complex would the migration be?\n마이그레이션 복잡도는 어때요?", ts: 14000 },
            { role: "agent", sender: "helper", text: "pm: Check nesting depth first - flat documents migrate easily\npm: 중첩 깊이부터 확인하세요 - 평면 문서는 쉽게 이전돼요", ts: 18000, noTimingTs: 14100 },
            { role: "agent", sender: "helper", text: "Deeply nested arrays need careful denormalization planning\n깊게 중첩된 배열은 비정규화 계획이 필요해요", ts: 21000, noTimingTs: 14200 },
            { role: "agent", sender: "helper", text: "analyst: For your monthly reports, PostgreSQL materialized views would be ideal\nanalyst: 월간 리포트엔 PostgreSQL materialized view가 이상적이에요", ts: 24000, noTimingTs: 14300 },
        ],
        baselineMessages: [
            { role: "agent", sender: "helper", text: "MongoDB and PostgreSQL serve different use cases. MongoDB is a NoSQL document database that stores data in JSON-like documents, offering flexible schemas. PostgreSQL is a relational database that uses tables with predefined schemas and supports complex queries with SQL. MongoDB is better for unstructured data and rapid development, while PostgreSQL excels at complex transactions and data integrity. The choice depends on your data structure, query patterns, and scalability requirements.\nMongoDB와 PostgreSQL은 다른 용도로 사용됩니다. MongoDB는 JSON 형태 문서를 저장하는 NoSQL 문서 데이터베이스로 유연한 스키마를 제공합니다. PostgreSQL은 미리 정의된 스키마의 테이블을 사용하고 SQL로 복잡한 쿼리를 지원하는 관계형 데이터베이스입니다.", ts: 9000, noTimingTs: 5100 },
            { role: "user", sender: "pm", text: "How complex would the migration be?\n마이그레이션 복잡도는 어때요?", ts: 14000 },
            { role: "agent", sender: "helper", text: "Migration complexity varies depending on your specific data and requirements.\n마이그레이션 복잡도는 특정 데이터와 요구사항에 따라 다릅니다.", ts: 18000, noTimingTs: 14100 },
        ]
    }
};

// Export for use in app.js and generate_gifs.py
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { SURVEY_SETS, GIF_CONVERSATIONS };
}
