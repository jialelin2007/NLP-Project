from __future__ import annotations

import gzip
import hashlib
import json
import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import pandas as pd

SYSTEM_PROMPT = (
    "You are a professional academic translator. Translate English CS/AI paper text "
    "into accurate, fluent, formal Chinese. Preserve technical terms, equations, "
    "citations, code, variable names, and LaTeX syntax. Do not add explanations."
)

USER_PROMPT_PREFIX = "Translate the following English academic text into Chinese:\n\n"

SELF_TALK_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"<\s*/?\s*think\s*>",
        r"\b(?:as an ai|i cannot|i can't|i am unable)\b",
        r"作为\s*(?:一个)?\s*ai",
        r"我是\s*(?:一个)?\s*ai",
        r"我无法",
    ]
]

HAN_RE = re.compile(r"[\u4e00-\u9fff]")
NON_CHINESE_SCRIPT_RE = re.compile(
    r"[\u3040-\u30ff\uac00-\ud7af\u0400-\u04ff\u0370-\u03ff\u0590-\u05ff"
    r"\u0600-\u06ff\u0900-\u097f\u0e00-\u0e7f]"
)
TRADITIONAL_ONLY_RE = re.compile(
    r"["
    r"萬與專業東絲丟兩嚴喪個豐臨為麗舉麼義烏樂喬習鄉書買亂爭於虧雲亞產畝親"
    r"褻嚲億僅從侖倉儀們價眾優夥會傘偉傳傷倫偽體餘佈來侶俠倖倀倆倉個們倫"
    r"偉側偵偽傑傖傘備傢傭債傾僂僅僑僕僞僥僨價儀儂億儈儉儐儔儕償優儲儷儼"
    r"兒兌兗黨蘭關興兹養獸內岡冊寫軍農冪凍凈淒準幾鳳凱別刪剄則剋剎劃劇劉"
    r"劍劑勁動務勛勝勞勢勻匭匯區協單賣盧鹵卻厙廠廳歷厲壓厭厙廁釐廂厴廈廚"
    r"廄縣參雙發變敘葉號嘆後嚇呂嗎啟吳員咼響啞問啓嘔喚喪喬單啞嗆嗇嗎嗚嘩"
    r"噦嘖嗩嘮嘗嘆嘍嘔嘯囑囪圍園圓圖團聖場壞塊堅壇壩塢墳墜壟壠壢壘墾壇壩"
    r"壯聲壹處備復夠夢夾奐奧奩奪奮奬妝婦媽嫵妳姍姦娛婁婦媧媯嬈嬋嬌嬤孫學"
    r"孿寧寶實寵審寫寬寢對尋導將專尷屆屍層屜屬岡峴島峽崍崑崗崙崢嶄嶇嶔嶁"
    r"嶗嶠嶧嶮嶸嶺嶼巒巔幣帥師帳帶幀幫幬幹幾庫廁廂廄廈廚廝廟廠廡廢廣廩廬"
    r"廳弒弔張彌彎彈強彆彙彥後徑從徠復徵徹恆恥悅悞悵悶惡惱惲惻愛愜愨愴愷"
    r"愾慄態慍慘慚慟慣慪慫慮慳慶憂憊憐憑憒憚憤憫憮憲憶懇應懌懍懣懲懶懷懸"
    r"懺懼懾戀戇戔戧戩戰戲戶拋挾捨捫掃掄掗掙掛採揀揚換揮損搖搗搶摑摜摟摯"
    r"摳摶摻撈撐撓撟撣撥撫撲撳撻撾撿擁擄擇擊擋擔據擠擬擯擰擱擲擴擷擺擻擼"
    r"擾攄攆攔攖攙攜攝攢攤攪攬敗敘敵數斂斃斕鬥斬斷於時曠暢暫曄曆曇曉曖曠"
    r"曨會朧東棄桿柵標棧棟欄樹樣橋機橫檢樓槳樂樅構槍槤槨槧槳樁樞標樞樸樺"
    r"樹樺橈橋橢機橢檔檢檣檜櫃櫓櫚櫛櫝櫞櫟櫥櫧櫨櫪櫫櫬櫱櫳櫸櫻權欄欅歡"
    r"歐歎殘殞殤殫殮殯殲殺殼毀毆毿氈氌氣氫氬氳汙決沒沖況洶浹涇涼淒淚淥淨"
    r"淩淪淵淶淺渙減湊湞湯溈準溝溫滄滅滌滎滬滯滲滸滻滾滿漁漚漢漣漬漲漵漸"
    r"漿潁潑潔潛潤潯潰潷潿澀澆澇澗澠澤澦澩濁濃濕濘濟濤濫濰濱濺濼濾瀅瀆瀉"
    r"瀋瀏瀕瀘瀝瀟瀠瀦瀧瀨瀰瀲瀾灃灄灑灘灣灤灧災為烏烴無煉煒煙煢煥煩煬熱"
    r"熾燁燈燉燒燙燜營燦燭燴燼燾爍爐爛爭爲爺爾牆牘牽犖犢犧狀狹狽猙猶猻獁"
    r"獃獄獅獎獨獪獫獮獰獲獵獷獸獺獻獼玀現琺琿瑋瑣瑤瑩瑪瑲璉璣璦環璽瓊瓏"
    r"瓔瓚甌產畢異畫當疇疊痙痾瘂瘋瘍瘓瘞瘡瘧瘮瘺瘻療癆癇癉癘癟癢癤癥癧癩"
    r"癬癭癮癰癱癲發皚皺盃盜盞盡監盤盧盪眥眾睏睜睞瞞瞭瞼矇矯硃硤硨確碼磚"
    r"磣磧磯礎礙礦礪礫礬祿禍禎禦禪禮禰禱禿稈稅稜稟種稱穀穌積穎穩窩窪窮竄"
    r"竅竇竊競筆筍筧箋箏節範築篋篔篤篩篳簀簍簞簡簣簫簷簽簾籃籌籙籜籟籠籩"
    r"籪籬粵糝糞糧糲糴糶糾紀紂約紅紆紇紈紉紋納紐紓純紕紗紙級紛紜紡紮細紱紲"
    r"紳紵紹紺紿絀終組絆絎結絕絛絞絡絢給絨絰統絲絳絹綁綃綆綈綉綌綏綐經綜綠"
    r"綢綣綫綬維綯綰綱網綴綵綸綹綺綻綽綾綿緄緇緊緋緒緔緗緘緙線緝緞締緡緣"
    r"緦編緩緬緯緱緲練緶緹縈縉縊縋縐縑縕縗縛縝縞縟縣縧縫縭縮縱縲縴縵縶縷"
    r"縹總績繃繅繆繒織繕繚繞繡繢繩繪繫繭繮繯繰繳繹繼繽纈纊續纍纏纓纖纜缽"
    r"罈罌罰罵罷羅羆羈羋羥義習翹聖聞聯聰聲聳職聶聹聽聾肅脅脈脛脫脹腎腖腡"
    r"腦腫腳腸膃膚膠膩膽膾膿臉臍臏臘臚臟臠臢臨臺與興舉艙艦艫艱艷藝節芻茲"
    r"荊莊莖莢華萇萊萬萵葉葒葦葷蒔蒞蒼蓀蓋蓮蓯蓴蓽蔔蔞蔣蔥蔦蔭蕁蕆蕎蕒"
    r"蕓蕕蕘蕢蕩蕪蕭蕷薈薊薌薔薘薟薦薩薺藍藎藝藥藪藶藹藺蘄蘆蘇蘊蘋蘚蘞蘢"
    r"蘭蘺蘿處虛虜號虧蟲虯蛺蛻蜆蝕蝟蝦蝸螄螞螢螻螿蟄蟈蟎蟣蟬蟯蟲蟶蟻蠅蠆"
    r"蠍蠐蠑蠔蠟蠣蠱蠶蠻術衛衝袞裊裏補裝製裡複褲褳褸褻襇襖襝襠襤襪襬襯襲"
    r"見觀規覓視覘覡覥覦親覬覯覲覷覺覽覿觀觴觶觸訁訂訃計訊訌討訐訓訕訖託"
    r"記訛訝訟訢訣訥訪設許訴訶診註詁詆詎詐詒詔評詖詗詘詛詞詠詡詢詣試詩詫"
    r"詬詭詮詰話該詳詵詼詿誄誅誆誇誌認誑誒誕誘誚語誠誡誣誤誥誦誨說説誰課"
    r"誶誹誼誾調諂諄談諉請諍諏諑諒論諗諛諜諝諞諠諡諢諤諦諧諫諭諮諱諳諶諷"
    r"諸諺諼諾謀謁謂謄謅謊謎謐謔謖謗謙謚講謝謠謡謨謫謬謭謳謹謾譁證譎譏譖"
    r"識譙譚譜譫譯議譴護譸譽讀變讎讒讓讕讖讚讞豈豎豐豬貓貝貞負財貢貧貨販"
    r"貪貫責貯貰貲貳貴貶買貸貺費貼貽貿賀賁賂賃賄資賈賊賑賒賓賕賙賚賜賞賠"
    r"賡賢賣賤賦質賬賭賴賺賻購賽賾贄贅贈贊贋贍贏贓贔贖贛赬趕趙趨趲跡踐踴"
    r"蹌蹕蹣蹤蹺躂躉躊躋躍躑躒躓躚躡躥躦車軋軌軍軒軔軛軟軫軲軸軹軺軻軼軾"
    r"較輅載輊輒輔輕輛輜輝輞輟輥輦輪輯輳輸輻輾輿轀轂轄轅轆轉轍轎轔轟轡轢"
    r"轤辦辭辯農逕這連週進遊運過達違遙遜遞遠適遲遷選遺遼邁還邇邊邏邐郟郵"
    r"鄆鄉鄒鄔鄖鄧鄭鄰鄲鄴鄶鄺酇醜醞醫醬釀釁釃釅釋釐釒釓釔釕釗釘釙針釣釤"
    r"釦釧釩釵釷釹釺鈀鈁鈃鈄鈈鈉鈍鈎鈐鈑鈔鈕鈞鈣鈥鈦鈧鈮鈰鈳鈴鈷鈸鈹鈺鈽"
    r"鈾鈿鉀鉅鉈鉉鉋鉍鉑鉕鉗鉚鉛鉞鉢鉤鉦鉬鉭鉯鉸鉺鉻鉿銀銃銅銑銓銖銘銚銛"
    r"銜銠銣銥銦銨銩銪銫銬銱銳銷銹銻銼鋁鋃鋅鋇鋈鋌鋏鋒鋙鋟鋣鋤鋥鋦鋨鋩鋪"
    r"鋮鋯鋰鋱鋶錄錆錇錈錐錒錕錘錙錚錛錟錠錡錢錦錨錫錮錯錳錶鍀鍁鍃鍆鍇鍈"
    r"鍉鍋鍍鍔鍘鍚鍛鍞鍠鍤鍥鍩鍬鍯鍰鍵鍶鍺鍼鎂鎄鎇鎊鎌鎔鎖鎘鎛鎝鎡鎢鎣鎦"
    r"鎧鎩鎪鎬鎮鎰鎲鎳鎵鎸鎿鏃鏈鏇鏌鏍鏑鏗鏘鏜鏝鏞鏟鏡鏢鏤鏨鏰鏵鏷鏹鏺鏻"
    r"鐃鐋鐐鐒鐓鐔鐘鐙鐠鐦鐧鐨鐫鐮鐲鐳鐵鐶鐸鐺鑄鑊鑌鑑鑒鑞鑠鑣鑤鑥鑪鑭鑰"
    r"鑲鑷鑹鑼鑽鑾鑿長門閂閃閆閉開閌閎閏閑間閔閘閡閣閤閥閨閩閫閬閭閱閶閹"
    r"閻閼閽閾闃闆闈闊闋闌闍闐闔闕闖關闞闡闢闤阨阪陘陝陣陰陳陸陽隉隊階際"
    r"隕際隨險隱隴隸隻雋雖雙雛雜雞離難雲電霧霽靂靄靈靚靜靦鞏鞦鞽韁韃韆韉"
    r"韋韌韓韙韜韞韻響頁頂頃項順須頊頌預頑頒頓頗領頜頡頤頦頭頰頲頴頷頸頻"
    r"顆題額顎顏顓願顙顛類顢顧顫顯顰顱風颭颮颯颱颳颶颸颺颼飀飄飆飛飠飢飣"
    r"飥飩飪飫飭飯飲飴飼飽飾餃餄餅餉養餌餎餏餑餒餓餘餚餛餜餞餡館餭餱餳餵"
    r"餶餷餾餿饁饃饅饈饉饋饌饑饒饗饜饞饢馬馭馮馱馳馴駁駐駑駒駔駕駘駙駛駝"
    r"駟駡駢駭駮駰駱駸駿騁騅騍騎騏騖騙騤騫騭騮騰騶騷騸騾驀驁驂驃驄驅驊"
    r"驍驏驕驗驚驛驟驢驣驤驥驦驪驫骯髏髒體髕髖鬆鬍鬚鬥鬧鬩鬮鬱魎魘魚魛"
    r"魟魢魤魨魯魴魷鮁鮃鮊鮋鮍鮎鮐鮑鮒鮓鮚鮞鮣鮤鮦鮪鮫鮭鮮鮺鯀鯁鯇鯉鯊"
    r"鯒鯔鯖鯗鯛鯝鯡鯢鯤鯧鯨鯪鯫鯰鯴鯷鯽鯿鰁鰂鰃鰈鰉鰌鰍鰒鰓鰜鰟鰠鰣"
    r"鰥鰨鰩鰭鰮鰱鰲鰳鰵鰷鰹鰺鰻鰼鰾鱂鱅鱈鱉鱒鱔鱖鱗鱘鱝鱟鱠鱣鱤鱧鱨"
    r"鱭鱯鱷鱸鱺鳥鳧鳩鳲鳳鳴鳶鴆鴇鴉鴒鴕鴛鴝鴟鴣鴦鴨鴯鴰鴴鴻鴿鵂鵃鵐鵑"
    r"鵒鵓鵜鵝鵠鵡鵪鵬鵮鵯鵲鶇鶉鶓鶘鶚鶥鶩鶯鶲鶴鶺鷂鷄鷈鷓鷗鷙鷚鷥鷦"
    r"鷯鷲鷸鷺鷹鸌鸏鸚鸛鸝鹵鹹鹺鹼鹽麗麥麩黃黌點黨黲黴黷鼇鼉齊齋齎齏齒"
    r"齔齕齙齜齟齠齡齦齧齪齬齲齶齷龍龐龔龕龜"
    r"]"
)


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def join_text_parts(*parts: Any) -> str:
    cleaned = [normalize_text(part) for part in parts]
    return "\n\n".join(part for part in cleaned if part)


def contains_self_talk(text: str) -> bool:
    return any(pattern.search(text) for pattern in SELF_TALK_PATTERNS)


def contains_non_chinese_script(text: str) -> bool:
    return bool(NON_CHINESE_SCRIPT_RE.search(text))


def contains_traditional_chinese(text: str) -> bool:
    return bool(TRADITIONAL_ONLY_RE.search(text))


def simplified_chinese_ratio(text: str) -> float:
    if not text:
        return 0.0
    return len(HAN_RE.findall(text)) / len(text)


def is_simplified_chinese_target(text: str, *, min_han_ratio: float = 0.2) -> bool:
    text = normalize_text(text)
    if simplified_chinese_ratio(text) < min_han_ratio:
        return False
    if contains_non_chinese_script(text):
        return False
    return not contains_traditional_chinese(text)


def is_valid_translation_pair(
    source: str,
    target: str,
    *,
    min_source_chars: int = 12,
    min_target_chars: int = 6,
    max_char_ratio: float = 8.0,
) -> bool:
    source = normalize_text(source)
    target = normalize_text(target)
    if len(source) < min_source_chars or len(target) < min_target_chars:
        return False
    if contains_self_talk(source) or contains_self_talk(target):
        return False
    if not is_simplified_chinese_target(target):
        return False
    ratio = max(len(source), len(target)) / max(1, min(len(source), len(target)))
    return ratio <= max_char_ratio


def make_quickmt_example(record: dict[str, Any], *, split: str, index: int) -> dict[str, Any]:
    source = normalize_text(record.get("en"))
    target = normalize_text(record.get("zh"))
    return {
        "id": f"quickmt_{split}_{index:010d}",
        "source": source,
        "target": target,
        "domain": "general",
        "split": split,
        "metadata": {
            "source_dataset": "quickmt",
            "score": record.get("sco"),
            "paper_id": None,
            "section": None,
        },
    }


def make_csl_example(
    zh_record: dict[str, Any], en_record: dict[str, Any], *, split: str
) -> dict[str, Any]:
    zh_doc_id = zh_record.get("doc_id")
    en_doc_id = en_record.get("doc_id")
    if zh_doc_id != en_doc_id:
        raise ValueError(f"CSL doc_id mismatch: {zh_doc_id!r} != {en_doc_id!r}")

    source = join_text_parts(en_record.get("title"), en_record.get("abstract"))
    target = join_text_parts(zh_record.get("title"), zh_record.get("abstract"))
    return {
        "id": str(zh_doc_id),
        "source": source,
        "target": target,
        "domain": "scientific",
        "split": split,
        "metadata": {
            "source_dataset": "csl",
            "paper_id": zh_doc_id,
            "section": "title_abstract",
            "keywords": zh_record.get("keywords") or [],
            "keywords_en": en_record.get("keywords") or [],
            "category": zh_record.get("category"),
            "category_eng": zh_record.get("category_eng") or en_record.get("category"),
            "discipline": zh_record.get("discipline"),
            "discipline_eng": zh_record.get("discipline_eng") or en_record.get("discipline"),
        },
    }


def build_sft_record(example: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": example["id"],
        "domain": example["domain"],
        "split": example["split"],
        "metadata": example["metadata"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{USER_PROMPT_PREFIX}{example['source']}"},
            {"role": "assistant", "content": example["target"]},
        ],
    }


def split_by_stable_hash(identifier: str, *, validation_pct: int = 10, test_pct: int = 10) -> str:
    bucket = int(hashlib.sha1(identifier.encode("utf-8")).hexdigest(), 16) % 100
    if bucket < validation_pct:
        return "validation"
    if bucket < validation_pct + test_pct:
        return "test"
    return "train"


def iter_jsonl_gz(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_quickmt_parquet(path: Path) -> Iterator[dict[str, Any]]:
    table = pd.read_parquet(path)
    yield from table.to_dict(orient="records")
