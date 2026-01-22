/**
 * Survey Data - 10 Comparison Sets
 *
 * Mock conversations for various topics
 * Each message has English text + Korean translation
 *
 * Structure:
 * - Sets 1-4: Full vs Baseline (text)
 * - Sets 5-6: Full vs NoTiming (GIF)
 * - Sets 7-8: Full vs NoTopicalFit (text)
 * - Sets 9-10: Full vs NoChunking (GIF)
 *
 * Topics:
 * 1. Ubuntu (package installation)
 * 2. Python (pip conflict)
 * 3. Python (virtual environment)
 * 4. Office work (Excel formula)
 * 5. Office work (email writing) - GIF
 * 6. Finance (stock terms) - GIF
 * 7. Finance (ETF recommendation)
 * 8. LLM usage (prompt writing)
 * 9. LLM usage (API integration) - GIF
 * 10. LLM usage (token optimization) - GIF
 */

const SURVEY_SETS = [
    // ============================================
    // Sets 1-4: Full vs Baseline
    // ============================================

    // Set 1: Ubuntu - package installation
    {
        type: "text",
        comparisonType: "Baseline",
        topic: "Ubuntu package installation",
        full: [
            { role: "user", sender: "alex", text: "How do I install Chrome on Ubuntu? The software center only shows Chromium.\n우분투에서 크롬 설치하려면 어떻게 해요? 소프트웨어 센터에는 크로미움만 있네요.", time: "14:23" },
            { role: "user", sender: "maya", text: "Also having issues with my apt - it says some packages are held back\n저도 apt 문제가 있어요 - 일부 패키지가 보류됐다고 나와요", time: "14:24" },
            { role: "user", sender: "jordan", text: "Chrome isn't in the repos, you need the .deb from Google\n크롬은 저장소에 없어요, 구글에서 .deb 파일 받아야 해요", time: "14:25" },
            { role: "agent", sender: "helper", text: "alex: Download from google.com/chrome\nalex: google.com/chrome에서 다운로드하세요", time: "14:26" },
            { role: "agent", sender: "helper", text: "Then run: sudo dpkg -i google-chrome*.deb\n그다음 실행: sudo dpkg -i google-chrome*.deb", time: "14:26" },
            { role: "user", sender: "alex", text: "Got it, downloading now\n알겠어요, 지금 다운로드 중이에요", time: "14:27" },
            { role: "agent", sender: "helper", text: "maya: Run sudo apt update first\nmaya: 먼저 sudo apt update 실행해보세요", time: "14:28" },
            { role: "agent", sender: "helper", text: "Then try sudo apt upgrade\n그다음 sudo apt upgrade 해보세요", time: "14:28" },
            { role: "user", sender: "maya", text: "The held back packages are still there after update\n업데이트 후에도 보류된 패키지가 그대로예요", time: "14:30" },
            { role: "agent", sender: "helper", text: "maya: Those are phased updates, just wait a few days\nmaya: 단계적 업데이트예요, 며칠 기다리면 돼요", time: "14:31" },
        ],
        comparison: [
            { role: "user", sender: "alex", text: "How do I install Chrome on Ubuntu? The software center only shows Chromium.\n우분투에서 크롬 설치하려면 어떻게 해요? 소프트웨어 센터에는 크로미움만 있네요.", time: "14:23" },
            { role: "user", sender: "maya", text: "Also having issues with my apt - it says some packages are held back\n저도 apt 문제가 있어요 - 일부 패키지가 보류됐다고 나와요", time: "14:24" },
            { role: "user", sender: "jordan", text: "Chrome isn't in the repos, you need the .deb from Google\n크롬은 저장소에 없어요, 구글에서 .deb 파일 받아야 해요", time: "14:25" },
            { role: "agent", sender: "helper", text: "For Chrome installation: Google Chrome is not available in Ubuntu's official repositories because it's proprietary software. You need to download the .deb package directly from google.com/chrome. Once downloaded, install it using 'sudo dpkg -i google-chrome-stable_current_amd64.deb'. If there are dependency errors, run 'sudo apt -f install' to fix them. For the held back packages issue: This is normal behavior called 'phased updates' where Ubuntu gradually rolls out updates to detect issues. Run 'sudo apt update' followed by 'sudo apt upgrade'. If packages are still held back, you can either wait a few days or force the upgrade with 'sudo apt full-upgrade', though waiting is recommended.\n크롬 설치: 구글 크롬은 독점 소프트웨어라 우분투 공식 저장소에 없습니다. google.com/chrome에서 .deb 패키지를 직접 다운로드해야 합니다. 다운로드 후 'sudo dpkg -i google-chrome-stable_current_amd64.deb'로 설치하세요. 의존성 오류가 있으면 'sudo apt -f install'로 해결하세요. 보류된 패키지 문제: 이것은 우분투가 문제를 감지하기 위해 점진적으로 업데이트를 배포하는 '단계적 업데이트'라는 정상적인 동작입니다. 'sudo apt update' 후 'sudo apt upgrade'를 실행하세요. 여전히 보류되면 며칠 기다리거나 'sudo apt full-upgrade'로 강제 업그레이드할 수 있지만, 기다리는 것을 권장합니다.", time: "14:25" },
            { role: "user", sender: "alex", text: "Got it, downloading now\n알겠어요, 지금 다운로드 중이에요", time: "14:27" },
            { role: "user", sender: "maya", text: "The held back packages are still there after update\n업데이트 후에도 보류된 패키지가 그대로예요", time: "14:30" },
            { role: "agent", sender: "helper", text: "As I mentioned, phased updates are normal. Just wait a few days.\n말씀드렸듯이, 단계적 업데이트는 정상입니다. 며칠만 기다리세요.", time: "14:31" },
        ]
    },

    // Set 2: Python - pip conflict
    {
        type: "text",
        comparisonType: "Baseline",
        topic: "Python pip dependency conflict",
        full: [
            { role: "user", sender: "sam", text: "pip install keeps failing with dependency conflicts between numpy and pandas\npip install이 numpy랑 pandas 의존성 충돌로 계속 실패해요", time: "10:15" },
            { role: "user", sender: "taylor", text: "What versions are you trying to install?\n어떤 버전을 설치하려고 해요?", time: "10:16" },
            { role: "user", sender: "sam", text: "numpy 1.21 and pandas 2.0\nnumpy 1.21이랑 pandas 2.0이요", time: "10:17" },
            { role: "agent", sender: "helper", text: "sam: pandas 2.0 needs numpy >= 1.23\nsam: pandas 2.0은 numpy 1.23 이상이 필요해요", time: "10:18" },
            { role: "agent", sender: "helper", text: "Either upgrade numpy or use pandas 1.5\nnumpy를 업그레이드하거나 pandas 1.5를 쓰세요", time: "10:18" },
            { role: "user", sender: "riley", text: "I had the same issue last week\n저도 지난주에 같은 문제 있었어요", time: "10:19" },
            { role: "user", sender: "sam", text: "But my other package needs numpy < 1.22\n근데 다른 패키지가 numpy 1.22 미만을 필요로 해요", time: "10:20" },
            { role: "agent", sender: "helper", text: "sam: Then you need separate virtual environments\nsam: 그러면 가상환경을 분리해야 해요", time: "10:21" },
            { role: "agent", sender: "helper", text: "One for each incompatible project\n호환 안 되는 프로젝트마다 하나씩요", time: "10:21" },
            { role: "user", sender: "sam", text: "Makes sense, thanks!\n이해됐어요, 감사합니다!", time: "10:22" },
        ],
        comparison: [
            { role: "user", sender: "sam", text: "pip install keeps failing with dependency conflicts between numpy and pandas\npip install이 numpy랑 pandas 의존성 충돌로 계속 실패해요", time: "10:15" },
            { role: "user", sender: "taylor", text: "What versions are you trying to install?\n어떤 버전을 설치하려고 해요?", time: "10:16" },
            { role: "user", sender: "sam", text: "numpy 1.21 and pandas 2.0\nnumpy 1.21이랑 pandas 2.0이요", time: "10:17" },
            { role: "agent", sender: "helper", text: "The dependency conflict occurs because pandas 2.0 requires numpy version 1.23 or higher, but you're trying to install numpy 1.21. You have several options: 1) Upgrade numpy to 1.23+ with 'pip install numpy>=1.23 pandas==2.0', 2) Downgrade pandas to 1.5.x which supports older numpy with 'pip install numpy==1.21 pandas==1.5.3', 3) If you have conflicting requirements from different packages, create separate virtual environments for each project using 'python -m venv env_name'. You can also use pip's dependency resolver with 'pip install --use-feature=2020-resolver' for better conflict resolution. Check the exact requirements with 'pip show pandas' to see version constraints.\n의존성 충돌은 pandas 2.0이 numpy 1.23 이상을 요구하는데 numpy 1.21을 설치하려고 해서 발생합니다. 여러 옵션이 있습니다: 1) 'pip install numpy>=1.23 pandas==2.0'으로 numpy를 1.23+로 업그레이드, 2) 'pip install numpy==1.21 pandas==1.5.3'으로 이전 numpy를 지원하는 pandas 1.5.x로 다운그레이드, 3) 다른 패키지에서 충돌하는 요구사항이 있으면 'python -m venv env_name'으로 각 프로젝트별 가상환경을 만드세요. 'pip install --use-feature=2020-resolver'로 pip의 의존성 해결기를 사용할 수도 있습니다. 'pip show pandas'로 정확한 요구사항을 확인하세요.", time: "10:17" },
            { role: "user", sender: "riley", text: "I had the same issue last week\n저도 지난주에 같은 문제 있었어요", time: "10:19" },
            { role: "user", sender: "sam", text: "But my other package needs numpy < 1.22\n근데 다른 패키지가 numpy 1.22 미만을 필요로 해요", time: "10:20" },
            { role: "agent", sender: "helper", text: "As mentioned, use separate virtual environments for incompatible projects.\n말씀드렸듯이, 호환되지 않는 프로젝트는 별도 가상환경을 사용하세요.", time: "10:21" },
            { role: "user", sender: "sam", text: "Makes sense, thanks!\n이해됐어요, 감사합니다!", time: "10:22" },
        ]
    },

    // Set 3: Python - virtual environment
    {
        type: "text",
        comparisonType: "Baseline",
        topic: "Python virtual environment setup",
        full: [
            { role: "user", sender: "jamie", text: "How do I set up a virtual environment? I keep messing up my global packages\n가상환경 어떻게 설정해요? 전역 패키지를 계속 망치고 있어요", time: "15:40" },
            { role: "user", sender: "casey", text: "Are you on Windows or Mac?\n윈도우예요 맥이에요?", time: "15:41" },
            { role: "user", sender: "jamie", text: "Windows 11\n윈도우 11이요", time: "15:41" },
            { role: "agent", sender: "helper", text: "jamie: Run python -m venv myenv\njamie: python -m venv myenv 실행하세요", time: "15:42" },
            { role: "agent", sender: "helper", text: "Then activate with myenv\\Scripts\\activate\n그다음 myenv\\Scripts\\activate로 활성화", time: "15:42" },
            { role: "user", sender: "morgan", text: "Don't forget to add myenv to .gitignore!\nmyenv를 .gitignore에 추가하는 거 잊지 마세요!", time: "15:43" },
            { role: "user", sender: "jamie", text: "It says 'running scripts is disabled on this system'\n'이 시스템에서 스크립트 실행이 비활성화되어 있습니다'래요", time: "15:44" },
            { role: "agent", sender: "helper", text: "jamie: PowerShell execution policy issue\njamie: PowerShell 실행 정책 문제예요", time: "15:45" },
            { role: "agent", sender: "helper", text: "Run: Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned\n실행: Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned", time: "15:45" },
            { role: "user", sender: "jamie", text: "That worked! Thanks everyone\n됐어요! 다들 감사해요", time: "15:47" },
        ],
        comparison: [
            { role: "user", sender: "jamie", text: "How do I set up a virtual environment? I keep messing up my global packages\n가상환경 어떻게 설정해요? 전역 패키지를 계속 망치고 있어요", time: "15:40" },
            { role: "user", sender: "casey", text: "Are you on Windows or Mac?\n윈도우예요 맥이에요?", time: "15:41" },
            { role: "user", sender: "jamie", text: "Windows 11\n윈도우 11이요", time: "15:41" },
            { role: "agent", sender: "helper", text: "To set up a Python virtual environment on Windows 11: 1) Open PowerShell or Command Prompt, 2) Navigate to your project folder with 'cd your_project', 3) Create the virtual environment with 'python -m venv myenv' (myenv is the folder name), 4) Activate it with 'myenv\\Scripts\\activate' (PowerShell) or 'myenv\\Scripts\\activate.bat' (CMD). Your prompt will show (myenv) when activated. If you get an error about scripts being disabled, run 'Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned' in PowerShell as administrator. To deactivate, just type 'deactivate'. Always add the venv folder to .gitignore to avoid committing it. You can also use 'pip freeze > requirements.txt' to save your dependencies.\nWindows 11에서 Python 가상환경 설정: 1) PowerShell 또는 명령 프롬프트 열기, 2) 'cd your_project'로 프로젝트 폴더로 이동, 3) 'python -m venv myenv'로 가상환경 생성 (myenv는 폴더 이름), 4) 'myenv\\Scripts\\activate' (PowerShell) 또는 'myenv\\Scripts\\activate.bat' (CMD)로 활성화. 활성화되면 프롬프트에 (myenv)가 표시됩니다. 스크립트 비활성화 오류가 나면 PowerShell에서 관리자 권한으로 'Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned' 실행하세요. 비활성화는 'deactivate' 입력. venv 폴더는 항상 .gitignore에 추가하세요. 'pip freeze > requirements.txt'로 의존성을 저장할 수 있습니다.", time: "15:41" },
            { role: "user", sender: "morgan", text: "Don't forget to add myenv to .gitignore!\nmyenv를 .gitignore에 추가하는 거 잊지 마세요!", time: "15:43" },
            { role: "user", sender: "jamie", text: "It says 'running scripts is disabled on this system'\n'이 시스템에서 스크립트 실행이 비활성화되어 있습니다'래요", time: "15:44" },
            { role: "agent", sender: "helper", text: "As I mentioned, run the Set-ExecutionPolicy command to fix the script execution error.\n말씀드렸듯이, Set-ExecutionPolicy 명령어로 스크립트 실행 오류를 해결하세요.", time: "15:45" },
            { role: "user", sender: "jamie", text: "That worked! Thanks everyone\n됐어요! 다들 감사해요", time: "15:47" },
        ]
    },

    // Set 4: Office work - Excel formula
    {
        type: "text",
        comparisonType: "Baseline",
        topic: "Excel VLOOKUP formula",
        full: [
            { role: "user", sender: "chris", text: "My VLOOKUP keeps returning #N/A even though the value exists\nVLOOKUP이 값이 있는데도 계속 #N/A를 반환해요", time: "09:30" },
            { role: "user", sender: "pat", text: "Are you using exact match or approximate?\n정확히 일치로 하고 있어요, 근사치로 하고 있어요?", time: "09:31" },
            { role: "user", sender: "chris", text: "I think exact match? I have TRUE at the end\n정확히 일치인 것 같아요? 끝에 TRUE 넣었어요", time: "09:32" },
            { role: "agent", sender: "helper", text: "chris: TRUE means approximate match\nchris: TRUE는 근사 일치예요", time: "09:33" },
            { role: "agent", sender: "helper", text: "Use FALSE for exact match\n정확히 일치는 FALSE를 쓰세요", time: "09:33" },
            { role: "user", sender: "drew", text: "Also check for extra spaces in your data\n데이터에 여분의 공백이 있는지도 확인해보세요", time: "09:34" },
            { role: "user", sender: "chris", text: "Changed to FALSE but still not working\nFALSE로 바꿨는데 여전히 안 돼요", time: "09:35" },
            { role: "agent", sender: "helper", text: "chris: Try wrapping lookup value with TRIM()\nchris: 찾는 값을 TRIM()으로 감싸보세요", time: "09:36" },
            { role: "agent", sender: "helper", text: "=VLOOKUP(TRIM(A1),B:C,2,FALSE)\n=VLOOKUP(TRIM(A1),B:C,2,FALSE)", time: "09:36" },
            { role: "user", sender: "chris", text: "That was it! Hidden spaces were the problem\n그거였어요! 숨겨진 공백이 문제였네요", time: "09:38" },
        ],
        comparison: [
            { role: "user", sender: "chris", text: "My VLOOKUP keeps returning #N/A even though the value exists\nVLOOKUP이 값이 있는데도 계속 #N/A를 반환해요", time: "09:30" },
            { role: "user", sender: "pat", text: "Are you using exact match or approximate?\n정확히 일치로 하고 있어요, 근사치로 하고 있어요?", time: "09:31" },
            { role: "user", sender: "chris", text: "I think exact match? I have TRUE at the end\n정확히 일치인 것 같아요? 끝에 TRUE 넣었어요", time: "09:32" },
            { role: "agent", sender: "helper", text: "The VLOOKUP #N/A error with TRUE at the end is a common mistake. In VLOOKUP syntax =VLOOKUP(lookup_value, table_array, col_index, [range_lookup]), the last parameter TRUE means approximate match (data must be sorted), while FALSE means exact match. For exact matching, always use FALSE. Other common causes of #N/A: 1) Extra spaces - use TRIM() on your lookup value like =VLOOKUP(TRIM(A1),B:C,2,FALSE), 2) Different data types - numbers stored as text vs actual numbers, use VALUE() or TEXT() to convert, 3) The lookup value is in a column to the right of the return column - VLOOKUP only searches leftmost column, 4) Table array doesn't include the lookup column. Consider using INDEX/MATCH or XLOOKUP (Excel 365) for more flexibility.\nVLOOKUP에서 끝에 TRUE를 넣고 #N/A 오류가 나는 것은 흔한 실수입니다. VLOOKUP 문법 =VLOOKUP(찾는값, 테이블범위, 열번호, [범위검색])에서 마지막 매개변수 TRUE는 근사 일치(데이터 정렬 필요), FALSE는 정확히 일치입니다. 정확한 일치에는 항상 FALSE를 사용하세요. #N/A의 다른 원인: 1) 여분의 공백 - =VLOOKUP(TRIM(A1),B:C,2,FALSE)처럼 TRIM() 사용, 2) 다른 데이터 유형 - 텍스트로 저장된 숫자 vs 실제 숫자, VALUE() 또는 TEXT()로 변환, 3) 찾는 값이 반환 열의 오른쪽에 있음 - VLOOKUP은 맨 왼쪽 열만 검색, 4) 테이블 범위에 검색 열이 포함되지 않음. 더 유연하게 INDEX/MATCH 또는 XLOOKUP(Excel 365)을 고려하세요.", time: "09:32" },
            { role: "user", sender: "drew", text: "Also check for extra spaces in your data\n데이터에 여분의 공백이 있는지도 확인해보세요", time: "09:34" },
            { role: "user", sender: "chris", text: "Changed to FALSE but still not working\nFALSE로 바꿨는데 여전히 안 돼요", time: "09:35" },
            { role: "agent", sender: "helper", text: "As I mentioned, try TRIM() to remove hidden spaces.\n말씀드렸듯이, TRIM()으로 숨겨진 공백을 제거해보세요.", time: "09:36" },
            { role: "user", sender: "chris", text: "That was it! Hidden spaces were the problem\n그거였어요! 숨겨진 공백이 문제였네요", time: "09:38" },
        ]
    },

    // ============================================
    // Sets 5-6: Full vs NoTiming (GIF)
    // ============================================

    // Set 5: Office work - email writing
    {
        type: "gif",
        comparisonType: "NoTiming",
        topic: "professional email writing",
        full: {
            src: "data/gifs/timing_full_1.gif",
            description: "Messages appear with natural delays"
        },
        comparison: {
            src: "data/gifs/timing_notiming_1.gif",
            description: "Messages appear instantly"
        }
    },

    // Set 6: Finance - stock terms
    {
        type: "gif",
        comparisonType: "NoTiming",
        topic: "stock market terminology",
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
    // Sets 7-8: Full vs NoTopicalFit
    // ============================================

    // Set 7: Finance - ETF recommendation
    {
        type: "text",
        comparisonType: "NoTopicalFit",
        topic: "ETF investment advice",
        full: [
            { role: "user", sender: "kevin", text: "I'm new to investing. What's the difference between ETF and mutual fund?\n투자 초보인데요. ETF랑 뮤추얼펀드 차이가 뭐예요?", time: "11:20" },
            { role: "user", sender: "lisa", text: "Also wondering about expense ratios\n보수율도 궁금해요", time: "11:21" },
            { role: "user", sender: "mike", text: "ETFs trade like stocks, mutual funds don't\nETF는 주식처럼 거래되고, 뮤추얼펀드는 안 그래요", time: "11:22" },
            { role: "agent", sender: "helper", text: "kevin: ETFs trade on exchange in real-time\nkevin: ETF는 거래소에서 실시간으로 거래돼요", time: "11:23" },
            { role: "agent", sender: "helper", text: "Mutual funds settle at end of day price\n뮤추얼펀드는 하루 마감 가격으로 정산돼요", time: "11:23" },
            { role: "user", sender: "kevin", text: "Which one has lower fees usually?\n보통 어느 쪽이 수수료가 낮아요?", time: "11:24" },
            { role: "agent", sender: "helper", text: "lisa: Expense ratio is the annual fee percentage\nlisa: 보수율은 연간 수수료 비율이에요", time: "11:25" },
            { role: "agent", sender: "helper", text: "kevin: ETFs typically have lower expense ratios\nkevin: ETF가 보통 보수율이 더 낮아요", time: "11:26" },
            { role: "user", sender: "lisa", text: "What's considered a good expense ratio?\n좋은 보수율은 어느 정도예요?", time: "11:27" },
            { role: "agent", sender: "helper", text: "lisa: Under 0.2% is excellent for index ETFs\nlisa: 인덱스 ETF는 0.2% 미만이면 훌륭해요", time: "11:28" },
        ],
        comparison: [
            { role: "user", sender: "kevin", text: "I'm new to investing. What's the difference between ETF and mutual fund?\n투자 초보인데요. ETF랑 뮤추얼펀드 차이가 뭐예요?", time: "11:20" },
            { role: "user", sender: "lisa", text: "Also wondering about expense ratios\n보수율도 궁금해요", time: "11:21" },
            { role: "user", sender: "mike", text: "ETFs trade like stocks, mutual funds don't\nETF는 주식처럼 거래되고, 뮤추얼펀드는 안 그래요", time: "11:22" },
            { role: "agent", sender: "helper", text: "Investing can be a great way to build wealth!\n투자는 재산을 늘리는 좋은 방법이에요!", time: "11:23" },
            { role: "user", sender: "kevin", text: "Which one has lower fees usually?\n보통 어느 쪽이 수수료가 낮아요?", time: "11:24" },
            { role: "agent", sender: "helper", text: "Fees are an important consideration in finance.\n수수료는 금융에서 중요한 고려사항이에요.", time: "11:25" },
            { role: "agent", sender: "helper", text: "Many people start investing early in life.\n많은 사람들이 일찍 투자를 시작해요.", time: "11:26" },
            { role: "user", sender: "lisa", text: "What's considered a good expense ratio?\n좋은 보수율은 어느 정도예요?", time: "11:27" },
            { role: "agent", sender: "helper", text: "Financial literacy is very valuable!\n금융 지식은 매우 가치 있어요!", time: "11:28" },
        ]
    },

    // Set 8: LLM usage - prompt writing
    {
        type: "text",
        comparisonType: "NoTopicalFit",
        topic: "ChatGPT prompt engineering",
        full: [
            { role: "user", sender: "anna", text: "How do I get better responses from ChatGPT? Mine are always too generic\nChatGPT에서 더 좋은 응답을 얻으려면 어떻게 해요? 제 건 항상 너무 일반적이에요", time: "16:10" },
            { role: "user", sender: "ben", text: "Same here, it keeps giving me surface-level answers\n저도요, 계속 피상적인 답변만 줘요", time: "16:11" },
            { role: "user", sender: "chloe", text: "Have you tried being more specific in your prompts?\n프롬프트를 더 구체적으로 해봤어요?", time: "16:12" },
            { role: "agent", sender: "helper", text: "anna: Give context about your use case\nanna: 사용 목적에 대한 맥락을 제공하세요", time: "16:13" },
            { role: "agent", sender: "helper", text: "Include role, format, and constraints\n역할, 형식, 제약 조건을 포함하세요", time: "16:13" },
            { role: "user", sender: "anna", text: "Can you give an example?\n예시를 들어줄 수 있어요?", time: "16:14" },
            { role: "agent", sender: "helper", text: "ben: Try: 'Act as a senior developer reviewing code'\nben: 이렇게 해보세요: '코드를 리뷰하는 시니어 개발자 역할을 해줘'", time: "16:15" },
            { role: "agent", sender: "helper", text: "anna: Instead of 'write about X', try 'write a 500-word blog post about X for beginners, focusing on Y'\nanna: 'X에 대해 써줘' 대신 '초보자를 위한 X에 대한 500단어 블로그 글을 Y에 초점을 맞춰서 써줘'로 해보세요", time: "16:16" },
            { role: "user", sender: "anna", text: "Oh that makes sense, I'll try that!\n아 그렇구나, 해볼게요!", time: "16:17" },
        ],
        comparison: [
            { role: "user", sender: "anna", text: "How do I get better responses from ChatGPT? Mine are always too generic\nChatGPT에서 더 좋은 응답을 얻으려면 어떻게 해요? 제 건 항상 너무 일반적이에요", time: "16:10" },
            { role: "user", sender: "ben", text: "Same here, it keeps giving me surface-level answers\n저도요, 계속 피상적인 답변만 줘요", time: "16:11" },
            { role: "user", sender: "chloe", text: "Have you tried being more specific in your prompts?\n프롬프트를 더 구체적으로 해봤어요?", time: "16:12" },
            { role: "agent", sender: "helper", text: "AI technology has advanced rapidly in recent years!\nAI 기술이 최근 몇 년간 급격히 발전했어요!", time: "16:13" },
            { role: "user", sender: "anna", text: "Can you give an example?\n예시를 들어줄 수 있어요?", time: "16:14" },
            { role: "agent", sender: "helper", text: "Large language models are fascinating!\n대규모 언어 모델은 정말 흥미로워요!", time: "16:15" },
            { role: "agent", sender: "helper", text: "Many companies are using AI these days.\n요즘 많은 회사들이 AI를 사용하고 있어요.", time: "16:16" },
            { role: "user", sender: "anna", text: "Oh that makes sense, I'll try that!\n아 그렇구나, 해볼게요!", time: "16:17" },
        ]
    },

    // ============================================
    // Sets 9-10: Full vs NoChunking (GIF)
    // ============================================

    // Set 9: LLM usage - API integration
    {
        type: "gif",
        comparisonType: "NoChunking",
        topic: "OpenAI API integration",
        full: {
            src: "data/gifs/chunking_full_1.gif",
            description: "Response split into natural chunks"
        },
        comparison: {
            src: "data/gifs/chunking_nochunk_1.gif",
            description: "Single long message"
        }
    },

    // Set 10: LLM usage - token optimization
    {
        type: "gif",
        comparisonType: "NoChunking",
        topic: "LLM token cost optimization",
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
    // Set 5: Office work - email writing (timing comparison)
    timing_1: {
        messages: [
            { role: "user", sender: "emma", text: "How do I write a professional email asking for a deadline extension?\n마감 연장 요청하는 비즈니스 이메일 어떻게 써요?", ts: 0 },
            { role: "user", sender: "frank", text: "Also need help with follow-up emails to clients\n저도 클라이언트한테 팔로업 이메일 쓰는 거 도움 필요해요", ts: 2500 },
            { role: "user", sender: "grace", text: "Be polite but direct is my advice\n공손하지만 직접적으로 쓰라는 게 제 조언이에요", ts: 5000 },
            { role: "agent", sender: "helper", text: "emma: Start with appreciation for their time\nemma: 먼저 시간 내주신 것에 감사를 표하세요", ts: 9000, noTimingTs: 5100 },
            { role: "agent", sender: "helper", text: "Then briefly explain the situation\n그다음 상황을 간략히 설명하세요", ts: 11500, noTimingTs: 5200 },
            { role: "user", sender: "emma", text: "Should I give a specific new date?\n구체적인 새 날짜를 제시해야 할까요?", ts: 14000 },
            { role: "agent", sender: "helper", text: "emma: Yes, always propose a concrete alternative\nemma: 네, 항상 구체적인 대안을 제시하세요", ts: 18000, noTimingTs: 14100 },
            { role: "agent", sender: "helper", text: "frank: For follow-ups, reference the previous email\nfrank: 팔로업은 이전 이메일을 언급하세요", ts: 21000, noTimingTs: 14200 },
            { role: "user", sender: "frank", text: "How long should I wait before following up?\n팔로업하기 전에 얼마나 기다려야 해요?", ts: 24000 },
            { role: "agent", sender: "helper", text: "frank: 3-5 business days is standard\nfrank: 영업일 기준 3-5일이 표준이에요", ts: 28000, noTimingTs: 24100 },
        ]
    },

    // Set 6: Finance - stock terms (timing comparison)
    timing_2: {
        messages: [
            { role: "user", sender: "henry", text: "What does P/E ratio mean? I see it everywhere\nP/E 비율이 뭐예요? 어디서나 보여요", ts: 0 },
            { role: "user", sender: "iris", text: "And what's the difference between market cap and enterprise value?\n시가총액이랑 기업가치 차이는 뭐예요?", ts: 2500 },
            { role: "user", sender: "jack", text: "P/E is price to earnings, basically how expensive the stock is\nP/E는 주가수익비율, 기본적으로 주식이 얼마나 비싼지예요", ts: 5000 },
            { role: "agent", sender: "helper", text: "henry: P/E = Stock Price ÷ Earnings Per Share\nhenry: P/E = 주가 ÷ 주당순이익", ts: 9000, noTimingTs: 5100 },
            { role: "agent", sender: "helper", text: "Lower P/E might mean undervalued\n낮은 P/E는 저평가됐을 수 있어요", ts: 11500, noTimingTs: 5200 },
            { role: "user", sender: "henry", text: "What's a good P/E ratio to look for?\n어느 정도 P/E 비율이 좋은 거예요?", ts: 14000 },
            { role: "agent", sender: "helper", text: "henry: Depends on industry, but 15-25 is average\nhenry: 산업마다 다르지만, 15-25가 평균이에요", ts: 18000, noTimingTs: 14100 },
            { role: "agent", sender: "helper", text: "iris: Market cap = share price × shares outstanding\niris: 시가총액 = 주가 × 발행주식수", ts: 21000, noTimingTs: 14200 },
            { role: "agent", sender: "helper", text: "Enterprise value adds debt, subtracts cash\n기업가치는 부채를 더하고 현금을 빼요", ts: 24000, noTimingTs: 14300 },
            { role: "user", sender: "iris", text: "So enterprise value is more accurate for comparing?\n그러면 비교할 때 기업가치가 더 정확한 거예요?", ts: 27000 },
            { role: "agent", sender: "helper", text: "iris: Yes, especially for companies with different debt levels\niris: 네, 특히 부채 수준이 다른 회사들 비교할 때요", ts: 31000, noTimingTs: 27100 },
        ]
    },

    // Set 9: LLM API integration (chunking comparison)
    chunking_1: {
        messages: [
            { role: "user", sender: "kate", text: "How do I integrate OpenAI API into my Python app?\n내 Python 앱에 OpenAI API 어떻게 연동해요?", ts: 0 },
            { role: "user", sender: "leo", text: "I'm also trying to figure out streaming responses\n저도 스트리밍 응답 구현하려고 해요", ts: 3000 },
            { role: "user", sender: "mia", text: "Make sure you don't expose your API key!\nAPI 키 노출하지 않게 조심하세요!", ts: 5500 },
            { role: "agent", sender: "helper", text: "kate: First, pip install openai\nkate: 먼저, pip install openai", ts: 9000, chunkDelay: 2500 },
            { role: "agent", sender: "helper", text: "Store your API key in environment variable\nAPI 키는 환경변수에 저장하세요", ts: 11500, chunkDelay: 2500 },
            { role: "user", sender: "kate", text: "What's the basic code structure?\n기본 코드 구조가 어떻게 돼요?", ts: 14000 },
            { role: "agent", sender: "helper", text: "kate: Import openai, then client = OpenAI()\nkate: openai를 import하고, client = OpenAI()", ts: 18000, chunkDelay: 2500 },
            { role: "agent", sender: "helper", text: "Call client.chat.completions.create()\nclient.chat.completions.create() 호출", ts: 20500, chunkDelay: 2500 },
            { role: "agent", sender: "helper", text: "Pass model='gpt-4' and messages list\nmodel='gpt-4'와 messages 리스트 전달", ts: 23000, chunkDelay: 2500 },
            { role: "user", sender: "leo", text: "And for streaming?\n스트리밍은요?", ts: 26000 },
            { role: "agent", sender: "helper", text: "leo: Add stream=True parameter\nleo: stream=True 매개변수 추가", ts: 30000, chunkDelay: 2500 },
            { role: "agent", sender: "helper", text: "Then iterate over response chunks\n그다음 응답 청크를 반복하세요", ts: 32500, chunkDelay: 2500 },
        ],
        fullChunked: true,
        comparisonSingleMessage: "kate: To integrate OpenAI API into your Python app, first install the package with 'pip install openai'. Store your API key securely in an environment variable (never hardcode it). The basic structure is: import openai, create a client with client = OpenAI(), then call client.chat.completions.create() with model='gpt-4' and a messages list containing your conversation. For streaming responses, add stream=True to the create() call and iterate over the response chunks with a for loop. Example: for chunk in response: print(chunk.choices[0].delta.content). Remember to handle rate limits and errors appropriately.\nkate: Python 앱에 OpenAI API를 연동하려면, 먼저 'pip install openai'로 패키지를 설치하세요. API 키는 환경변수에 안전하게 저장하세요 (절대 하드코딩하지 마세요). 기본 구조는: openai를 import하고, client = OpenAI()로 클라이언트를 생성한 후, model='gpt-4'와 대화가 담긴 messages 리스트로 client.chat.completions.create()를 호출합니다. 스트리밍 응답은 create() 호출에 stream=True를 추가하고 for 루프로 응답 청크를 반복합니다. 예시: for chunk in response: print(chunk.choices[0].delta.content). 속도 제한과 에러를 적절히 처리하는 것을 잊지 마세요."
    },

    // Set 10: LLM token optimization (chunking comparison)
    chunking_2: {
        messages: [
            { role: "user", sender: "nina", text: "My OpenAI bill is getting expensive. How do I reduce token usage?\nOpenAI 비용이 너무 많이 나와요. 토큰 사용량 어떻게 줄여요?", ts: 0 },
            { role: "user", sender: "oscar", text: "Same problem here, especially with long conversations\n저도 같은 문제예요, 특히 긴 대화에서요", ts: 3000 },
            { role: "user", sender: "paul", text: "Have you tried using gpt-3.5-turbo instead of gpt-4?\ngpt-4 대신 gpt-3.5-turbo 써봤어요?", ts: 5500 },
            { role: "agent", sender: "helper", text: "nina: First, shorten your system prompts\nnina: 먼저, 시스템 프롬프트를 줄이세요", ts: 9000, chunkDelay: 3000 },
            { role: "agent", sender: "helper", text: "Remove unnecessary instructions\n불필요한 지시사항을 제거하세요", ts: 12000, chunkDelay: 3000 },
            { role: "user", sender: "nina", text: "What else can I do?\n다른 건 뭐가 있어요?", ts: 15000 },
            { role: "agent", sender: "helper", text: "nina: Summarize long conversation histories\nnina: 긴 대화 기록은 요약하세요", ts: 19000, chunkDelay: 3000 },
            { role: "agent", sender: "helper", text: "Only keep recent messages in context\n최근 메시지만 맥락에 유지하세요", ts: 22000, chunkDelay: 3000 },
            { role: "agent", sender: "helper", text: "oscar: Use max_tokens to limit response length\noscar: max_tokens로 응답 길이를 제한하세요", ts: 25000, chunkDelay: 3000 },
            { role: "user", sender: "oscar", text: "Does the model choice matter much?\n모델 선택이 많이 중요해요?", ts: 28000 },
            { role: "agent", sender: "helper", text: "oscar: gpt-3.5-turbo is 10-20x cheaper\noscar: gpt-3.5-turbo가 10-20배 저렴해요", ts: 32000, chunkDelay: 3000 },
            { role: "agent", sender: "helper", text: "Use gpt-4 only when quality is critical\ngpt-4는 품질이 중요할 때만 쓰세요", ts: 35000, chunkDelay: 3000 },
        ],
        fullChunked: true,
        comparisonSingleMessage: "nina: To reduce OpenAI API token usage and costs: 1) Shorten system prompts - remove verbose instructions and keep only essential guidance, 2) Summarize conversation history - instead of passing entire chat history, summarize older messages and only keep recent ones in full, 3) Set max_tokens parameter to limit response length, 4) Use gpt-3.5-turbo instead of gpt-4 when possible (10-20x cheaper, still good for many tasks), 5) Implement caching for repeated queries, 6) Use tiktoken library to count tokens before sending to estimate costs. For long conversations, consider implementing a sliding window that keeps only the last N messages, or periodically summarize the conversation and start fresh with the summary.\nnina: OpenAI API 토큰 사용량과 비용을 줄이려면: 1) 시스템 프롬프트 줄이기 - 장황한 지시사항을 제거하고 필수적인 안내만 유지, 2) 대화 기록 요약 - 전체 채팅 기록 대신 오래된 메시지를 요약하고 최근 것만 전체로 유지, 3) max_tokens 매개변수로 응답 길이 제한, 4) 가능하면 gpt-4 대신 gpt-3.5-turbo 사용 (10-20배 저렴, 많은 작업에 여전히 좋음), 5) 반복 쿼리에 캐싱 구현, 6) tiktoken 라이브러리로 전송 전 토큰 수를 세어 비용 추정. 긴 대화의 경우 마지막 N개 메시지만 유지하는 슬라이딩 윈도우를 구현하거나, 주기적으로 대화를 요약하고 요약으로 새로 시작하는 것을 고려하세요."
    }
};

// Export for use in app.js and generate_gifs.py
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { SURVEY_SETS, GIF_CONVERSATIONS };
}
