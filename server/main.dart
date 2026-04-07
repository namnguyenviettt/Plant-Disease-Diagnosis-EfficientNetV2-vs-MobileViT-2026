import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  SystemChrome.setSystemUIOverlayStyle(const SystemUiOverlayStyle(
    statusBarColor: Colors.transparent,
    statusBarIconBrightness: Brightness.dark,
  ));
  runApp(const MyApp());
}

// ============================================================
// MODELS
// ============================================================

enum MessageType { image, text, ai }

class ChatMessage {
  final MessageType type;
  final String content;
  final String? imagePath;
  final DateTime time;

  ChatMessage({
    required this.type,
    required this.content,
    this.imagePath,
    DateTime? time,
  }) : time = time ?? DateTime.now();

  Map<String, dynamic> toJson() => {
    'type': type.index,
    'content': content,
    'imagePath': imagePath,
    'time': time.toIso8601String(),
  };

  factory ChatMessage.fromJson(Map<String, dynamic> j) => ChatMessage(
    type: MessageType.values[j['type'] as int],
    content: j['content'] as String,
    imagePath: j['imagePath'] as String?,
    time: DateTime.parse(j['time'] as String),
  );
}

class Conversation {
  final String id;
  String? serverConvId;
  String title;
  final List<ChatMessage> messages;
  final DateTime createdAt;

  Conversation({
    required this.id,
    this.serverConvId,
    required this.title,
    required this.messages,
    DateTime? createdAt,
  }) : createdAt = createdAt ?? DateTime.now();

  Map<String, dynamic> toJson() => {
    'id': id,
    'serverConvId': serverConvId,
    'title': title,
    'messages': messages.map((m) => m.toJson()).toList(),
    'createdAt': createdAt.toIso8601String(),
  };

  factory Conversation.fromJson(Map<String, dynamic> j) => Conversation(
    id: j['id'] as String,
    serverConvId: j['serverConvId'] as String?,
    title: j['title'] as String,
    messages: (j['messages'] as List)
        .map((m) => ChatMessage.fromJson(m as Map<String, dynamic>))
        .toList(),
    createdAt: DateTime.parse(j['createdAt'] as String),
  );
}

// ============================================================
// APP ROOT
// ============================================================

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        fontFamily: 'sans-serif',
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF2D5016)),
        useMaterial3: true,
      ),
      home: const HomeScreen(),
    );
  }
}

// ============================================================
// COLORS & CONSTANTS
// ============================================================

class AppColors {
  static const bg = Color(0xFFF4F1E8);
  static const surface = Color(0xFFEAE5D4);
  static const card = Color(0xFFFFFFFF);
  static const primary = Color(0xFF2D5016);
  static const primaryLight = Color(0xFF4A7C28);
  static const accent = Color(0xFF8BC34A);
  static const accentGold = Color(0xFFD4A843);
  static const textDark = Color(0xFF1A2E0A);
  static const textMid = Color(0xFF4A5E35);
  static const textLight = Color(0xFF8A9E72);
  static const bubble = Color(0xFFE8F5D4);
  static const bubbleAI = Color(0xFFFFFFFF);
  static const divider = Color(0xFFD4CBAA);
}

// ============================================================
// HOME SCREEN
// ============================================================

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen>
    with SingleTickerProviderStateMixin {
  List<Conversation> conversations = [];
  late AnimationController _leafCtrl;
  late Animation<double> _leafAnim;

  @override
  void initState() {
    super.initState();
    _leafCtrl = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 3),
    )..repeat(reverse: true);
    _leafAnim = Tween<double>(begin: -0.05, end: 0.05).animate(
      CurvedAnimation(parent: _leafCtrl, curve: Curves.easeInOut),
    );
    loadConversations();
  }

  @override
  void dispose() {
    _leafCtrl.dispose();
    super.dispose();
  }

  Future<void> loadConversations() async {
    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getString('conversations');
    if (raw != null) {
      try {
        final list = jsonDecode(raw) as List;
        setState(() {
          conversations = list
              .map((e) => Conversation.fromJson(e as Map<String, dynamic>))
              .toList()
            ..sort((a, b) => b.createdAt.compareTo(a.createdAt));
        });
      } catch (_) {}
    }
  }

  Future<void> saveConversations() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(
      'conversations',
      jsonEncode(conversations.map((c) => c.toJson()).toList()),
    );
  }

  Future<void> createNewConversation() async {
    final conv = Conversation(
      id: DateTime.now().millisecondsSinceEpoch.toString(),
      title: 'Cuộc trò chuyện mới',
      messages: [],
    );
    setState(() => conversations.insert(0, conv));
    await saveConversations();
    if (!mounted) return;
    await Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_) => ChatScreen(
          conversation: conv,
          onUpdate: (updated) async {
            setState(() {
              final idx = conversations.indexWhere((c) => c.id == updated.id);
              if (idx != -1) conversations[idx] = updated;
            });
            await saveConversations();
          },
          onDelete: (id) async {
            setState(() => conversations.removeWhere((c) => c.id == id));
            await saveConversations();
          },
        ),
      ),
    );
    await loadConversations();
  }

  Future<void> openConversation(Conversation conv) async {
    await Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_) => ChatScreen(
          conversation: conv,
          onUpdate: (updated) async {
            setState(() {
              final idx = conversations.indexWhere((c) => c.id == updated.id);
              if (idx != -1) conversations[idx] = updated;
            });
            await saveConversations();
          },
          onDelete: (id) async {
            setState(() => conversations.removeWhere((c) => c.id == id));
            await saveConversations();
          },
        ),
      ),
    );
    await loadConversations();
  }

  String _formatDate(DateTime dt) {
    final now = DateTime.now();
    final diff = now.difference(dt);
    if (diff.inDays == 0) return 'Hôm nay';
    if (diff.inDays == 1) return 'Hôm qua';
    return '${dt.day}/${dt.month}/${dt.year}';
  }

  @override
  Widget build(BuildContext context) {
    final safePadding = MediaQuery.of(context).padding;

    return Scaffold(
      backgroundColor: AppColors.bg,
      body: Column(
        children: [
          // ---- HEADER ----
          Container(
            height: safePadding.top + 140,
            decoration: const BoxDecoration(
              color: AppColors.primary,
              borderRadius: BorderRadius.only(
                bottomLeft: Radius.circular(32),
                bottomRight: Radius.circular(32),
              ),
            ),
            child: Stack(
              children: [
                Positioned(
                  top: -30,
                  right: -20,
                  child: Container(
                    width: 140,
                    height: 140,
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      color: AppColors.primaryLight.withOpacity(0.3),
                    ),
                  ),
                ),
                Positioned(
                  bottom: 10,
                  left: -30,
                  child: Container(
                    width: 100,
                    height: 100,
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      color: AppColors.accent.withOpacity(0.15),
                    ),
                  ),
                ),
                Positioned(
                  top: safePadding.top + 20,
                  right: 24,
                  child: AnimatedBuilder(
                    animation: _leafAnim,
                    builder: (_, child) => Transform.rotate(
                      angle: _leafAnim.value,
                      child: child,
                    ),
                    child: const Text('🌿', style: TextStyle(fontSize: 48)),
                  ),
                ),
                Positioned(
                  bottom: 24,
                  left: 24,
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text(
                        'Plant Doctor',
                        style: TextStyle(
                          color: Colors.white,
                          fontSize: 28,
                          fontWeight: FontWeight.w800,
                          letterSpacing: -0.5,
                        ),
                      ),
                      const SizedBox(height: 4),
                      Text(
                        'AI chẩn đoán bệnh cây trồng',
                        style: TextStyle(
                          color: Colors.white.withOpacity(0.7),
                          fontSize: 13,
                          fontWeight: FontWeight.w400,
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),

          const SizedBox(height: 20),

          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 20),
            child: Row(
              children: [
                const Text(
                  'Cuộc trò chuyện',
                  style: TextStyle(
                    color: AppColors.textDark,
                    fontSize: 16,
                    fontWeight: FontWeight.w700,
                  ),
                ),
                const Spacer(),
                Text(
                  '${conversations.length} đoạn',
                  style: const TextStyle(
                    color: AppColors.textLight,
                    fontSize: 13,
                  ),
                ),
              ],
            ),
          ),

          const SizedBox(height: 12),

          Expanded(
            child: conversations.isEmpty
                ? _buildEmptyState()
                : ListView.builder(
              padding: EdgeInsets.only(
                left: 16,
                right: 16,
                bottom: safePadding.bottom + 80,
              ),
              itemCount: conversations.length,
              itemBuilder: (_, i) => _buildConvCard(conversations[i]),
            ),
          ),
        ],
      ),
      floatingActionButton: Padding(
        padding: EdgeInsets.only(bottom: safePadding.bottom),
        child: FloatingActionButton.extended(
          onPressed: createNewConversation,
          backgroundColor: AppColors.primary,
          foregroundColor: Colors.white,
          elevation: 4,
          icon: const Icon(Icons.add_rounded),
          label: const Text(
            'Trò chuyện mới',
            style: TextStyle(fontWeight: FontWeight.w600),
          ),
        ),
      ),
      floatingActionButtonLocation: FloatingActionButtonLocation.centerFloat,
    );
  }

  Widget _buildEmptyState() {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          const Text('🌱', style: TextStyle(fontSize: 64)),
          const SizedBox(height: 16),
          const Text(
            'Chưa có cuộc trò chuyện nào',
            style: TextStyle(
              color: AppColors.textMid,
              fontSize: 16,
              fontWeight: FontWeight.w500,
            ),
          ),
          const SizedBox(height: 8),
          const Text(
            'Nhấn + để bắt đầu chẩn đoán cây của bạn',
            style: TextStyle(color: AppColors.textLight, fontSize: 13),
          ),
        ],
      ),
    );
  }

  Widget _buildConvCard(Conversation conv) {
    final lastMsg = conv.messages.isNotEmpty ? conv.messages.last : null;
    final hasImage = conv.messages.any((m) => m.type == MessageType.image);

    return GestureDetector(
      onTap: () => openConversation(conv),
      child: Container(
        margin: const EdgeInsets.only(bottom: 10),
        padding: const EdgeInsets.all(14),
        decoration: BoxDecoration(
          color: AppColors.card,
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: AppColors.divider, width: 1),
          boxShadow: [
            BoxShadow(
              color: AppColors.primary.withOpacity(0.06),
              blurRadius: 8,
              offset: const Offset(0, 2),
            ),
          ],
        ),
        child: Row(
          children: [
            Container(
              width: 48,
              height: 48,
              decoration: BoxDecoration(
                color: hasImage
                    ? AppColors.accent.withOpacity(0.15)
                    : AppColors.bubble,
                borderRadius: BorderRadius.circular(12),
              ),
              child: Center(
                child: Text(
                  hasImage ? '🌿' : '💬',
                  style: const TextStyle(fontSize: 22),
                ),
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    conv.title,
                    style: const TextStyle(
                      color: AppColors.textDark,
                      fontSize: 14,
                      fontWeight: FontWeight.w700,
                    ),
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                  ),
                  const SizedBox(height: 3),
                  if (lastMsg != null)
                    Text(
                      lastMsg.type == MessageType.image
                          ? '📷 Đã gửi ảnh cây'
                          : lastMsg.content,
                      style: const TextStyle(
                        color: AppColors.textLight,
                        fontSize: 12,
                      ),
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                    ),
                ],
              ),
            ),
            const SizedBox(width: 8),
            Column(
              crossAxisAlignment: CrossAxisAlignment.end,
              children: [
                Text(
                  _formatDate(conv.createdAt),
                  style: const TextStyle(
                    color: AppColors.textLight,
                    fontSize: 11,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  '${conv.messages.length} tin',
                  style: const TextStyle(
                    color: AppColors.textLight,
                    fontSize: 11,
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

// ============================================================
// CHAT SCREEN
// ============================================================

class ChatScreen extends StatefulWidget {
  final Conversation conversation;
  final Future<void> Function(Conversation) onUpdate;
  final Future<void> Function(String) onDelete;

  const ChatScreen({
    super.key,
    required this.conversation,
    required this.onUpdate,
    required this.onDelete,
  });

  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  late Conversation _conv;
  final TextEditingController _ctrl = TextEditingController();
  final ScrollController _scrollCtrl = ScrollController();
  final ImagePicker _picker = ImagePicker();

  bool _isLoading = false;
  String _streamingText = '';
  bool _isStreaming = false;

  // Ảnh đang chờ gửi (sau khi chọn, trước khi confirm)
  XFile? _pendingImage;
  final TextEditingController _promptCtrl = TextEditingController();

  static const String _baseUrl = 'http://172.21.200.107:8000';

  @override
  void initState() {
    super.initState();
    _conv = widget.conversation;
    WidgetsBinding.instance.addPostFrameCallback((_) => _scrollToBottom());
  }

  @override
  void dispose() {
    _ctrl.dispose();
    _scrollCtrl.dispose();
    _promptCtrl.dispose();
    super.dispose();
  }

  void _scrollToBottom() {
    if (_scrollCtrl.hasClients) {
      _scrollCtrl.animateTo(
        _scrollCtrl.position.maxScrollExtent + 200,
        duration: const Duration(milliseconds: 300),
        curve: Curves.easeOut,
      );
    }
  }

  Future<void> _addMessage(ChatMessage msg) async {
    setState(() {
      _conv.messages.add(msg);
      if (_conv.messages.length == 1) {
        if (msg.type == MessageType.image) {
          _conv.title = 'Chẩn đoán bệnh cây 🌿';
        } else {
          final words = msg.content.split(' ');
          _conv.title =
              words.take(5).join(' ') + (words.length > 5 ? '...' : '');
        }
      }
    });
    await widget.onUpdate(_conv);
    WidgetsBinding.instance.addPostFrameCallback((_) => _scrollToBottom());
  }

  // ===== HIỆN MENU CHỌN NGUỒN ẢNH =====
  void _showImageSourceMenu() {
    if (_isLoading) return;
    showModalBottomSheet(
      context: context,
      backgroundColor: Colors.transparent,
      builder: (_) => _ImageSourceSheet(
        onCamera: () {
          Navigator.pop(context);
          _pickImage(ImageSource.camera);
        },
        onGallery: () {
          Navigator.pop(context);
          _pickImage(ImageSource.gallery);
        },
      ),
    );
  }

  // ===== CHỌN ẢNH =====
  Future<void> _pickImage(ImageSource source) async {
    if (_isLoading) return;
    final photo = await _picker.pickImage(source: source);
    if (photo == null) return;

    setState(() => _pendingImage = photo);
    _promptCtrl.clear();

    if (!mounted) return;
    await showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (ctx) => Padding(
        padding: EdgeInsets.only(
          bottom: MediaQuery.of(ctx).viewInsets.bottom,
        ),
        child: _ImagePromptSheet(
          imagePath: _pendingImage!.path,
          promptCtrl: _promptCtrl,
          onSend: () {
            Navigator.pop(ctx);
            _sendImage();
          },
          onCancel: () {
            Navigator.pop(ctx);
            setState(() => _pendingImage = null);
          },
        ),
      ),
    );
  }

  // ===== GỬI ẢNH KÈM PROMPT =====
  Future<void> _sendImage() async {
    if (_pendingImage == null) return;

    final photo = _pendingImage!;
    final customPrompt = _promptCtrl.text.trim();
    setState(() => _pendingImage = null);

    // Lưu message ảnh, content = custom prompt nếu có
    final imgMsg = ChatMessage(
      type: MessageType.image,
      content: customPrompt.isNotEmpty ? customPrompt : 'Ảnh lá cây',
      imagePath: photo.path,
    );
    await _addMessage(imgMsg);

    setState(() => _isLoading = true);

    try {
      var request = http.MultipartRequest(
        'POST',
        Uri.parse('$_baseUrl/detect'),
      );

      if (_conv.serverConvId != null) {
        request.fields['conversation_id'] = _conv.serverConvId!;
      }

      // Gửi prompt lên server — nếu rỗng thì server dùng prompt mặc định
      if (customPrompt.isNotEmpty) {
        request.fields['prompt'] = customPrompt;
      }

      request.files.add(
        await http.MultipartFile.fromPath('file', photo.path),
      );

      final response = await request.send().timeout(
        const Duration(seconds: 30),
        onTimeout: () => throw Exception('Kết nối quá chậm, thử lại nhé!'),
      );
      final res = await http.Response.fromStream(response);
      final data = jsonDecode(res.body) as Map<String, dynamic>;

      if (data['conversation_id'] != null) {
        setState(
                () => _conv.serverConvId = data['conversation_id'] as String);
      }

      final solution = data['solution'] as String? ?? 'Không có giải pháp';
      await _simulateStream(solution);
    } catch (e) {
      await _addMessage(ChatMessage(
        type: MessageType.ai,
        content:
        '⚠️ Không thể kết nối server. Kiểm tra lại IP và thử lại.\n\nLỗi: ${e.toString().replaceAll('Exception: ', '')}',
      ));
    } finally {
      setState(() => _isLoading = false);
    }
  }

  // ===== CHATBOT với streaming =====
  Future<void> _sendMessage() async {
    final text = _ctrl.text.trim();
    if (text.isEmpty || _isLoading) return;

    _ctrl.clear();
    await _addMessage(ChatMessage(type: MessageType.text, content: text));

    setState(() => _isLoading = true);

    if (_conv.serverConvId == null) {
      try {
        final newConvRes = await http.post(
          Uri.parse('$_baseUrl/conversation/new'),
        ).timeout(const Duration(seconds: 10));
        final newConvData =
        jsonDecode(newConvRes.body) as Map<String, dynamic>;
        setState(
                () => _conv.serverConvId = newConvData['conversation_id'] as String);
        await widget.onUpdate(_conv);
      } catch (_) {}
    }

    final payload = jsonEncode({
      'conversation_id': _conv.serverConvId ?? 'default',
      'question': text,
    });

    try {
      final request = http.Request('POST', Uri.parse('$_baseUrl/chat/stream'))
        ..headers['Content-Type'] = 'application/json'
        ..headers['Accept'] = 'text/event-stream'
        ..body = payload;

      final streamedResponse = await request.send().timeout(
        const Duration(seconds: 10),
        onTimeout: () => throw Exception('timeout'),
      );

      if (streamedResponse.statusCode == 200) {
        setState(() {
          _isStreaming = true;
          _streamingText = '';
        });

        final buffer = StringBuffer();
        bool done = false;

        await for (final chunk in streamedResponse.stream
            .transform(utf8.decoder)
            .timeout(const Duration(seconds: 60))) {
          for (final line in chunk.split('\n')) {
            if (line.startsWith('data: ')) {
              final raw = line.substring(6);
              if (raw.trimRight() == '[DONE]') {
                done = true;
                break;
              }
              try {
                final decoded = jsonDecode(raw) as String;
                buffer.write(decoded);
              } catch (_) {
                buffer.write(raw);
              }
              setState(() => _streamingText = buffer.toString());
              _scrollToBottom();
            }
          }
          if (done) break;
        }

        final finalText = buffer.toString();
        setState(() {
          _isStreaming = false;
          _streamingText = '';
        });

        if (finalText.isNotEmpty) {
          await _addMessage(
              ChatMessage(type: MessageType.ai, content: finalText));
        }
      } else {
        throw Exception('HTTP ${streamedResponse.statusCode}');
      }
    } catch (streamErr) {
      setState(() {
        _isStreaming = false;
        _streamingText = '';
      });

      try {
        final response = await http.post(
          Uri.parse('$_baseUrl/chat'),
          headers: {'Content-Type': 'application/json'},
          body: payload,
        ).timeout(const Duration(seconds: 30));

        final data = jsonDecode(response.body) as Map<String, dynamic>;
        final answer = data['answer'] as String? ?? 'Không có phản hồi';
        await _simulateStream(answer);
      } catch (e) {
        setState(() => _isStreaming = false);
        await _addMessage(ChatMessage(
          type: MessageType.ai,
          content:
          '⚠️ Lỗi kết nối: ${e.toString().replaceAll('Exception: ', '')}',
        ));
      }
    } finally {
      setState(() => _isLoading = false);
    }
  }

  Future<void> _simulateStream(String fullText) async {
    setState(() {
      _isStreaming = true;
      _streamingText = '';
    });
    WidgetsBinding.instance.addPostFrameCallback((_) => _scrollToBottom());

    final buffer = StringBuffer();
    for (int i = 0; i < fullText.length; i++) {
      buffer.write(fullText[i]);
      setState(() => _streamingText = buffer.toString());

      final char = fullText[i];
      int delay = 18;
      if (char == '.' || char == '!' || char == '?') delay = 80;
      else if (char == ',' || char == ':') delay = 40;
      else if (char == '\n') delay = 50;

      await Future.delayed(Duration(milliseconds: delay));

      if (i % 20 == 0) _scrollToBottom();
    }

    final finalText = buffer.toString();
    setState(() {
      _isStreaming = false;
      _streamingText = '';
    });

    await _addMessage(ChatMessage(type: MessageType.ai, content: finalText));
  }

  Future<void> _deleteConversation() async {
    final confirm = await showDialog<bool>(
      context: context,
      builder: (_) => AlertDialog(
        title: const Text('Xoá cuộc trò chuyện?'),
        content: const Text('Hành động này không thể hoàn tác.'),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context, false),
            child: const Text('Huỷ'),
          ),
          TextButton(
            onPressed: () => Navigator.pop(context, true),
            style: TextButton.styleFrom(foregroundColor: Colors.red),
            child: const Text('Xoá'),
          ),
        ],
      ),
    );
    if (confirm == true) {
      if (_conv.serverConvId != null) {
        try {
          await http
              .delete(
            Uri.parse('$_baseUrl/conversation/${_conv.serverConvId}'),
          )
              .timeout(const Duration(seconds: 5));
        } catch (_) {}
      }
      await widget.onDelete(_conv.id);
      if (mounted) Navigator.pop(context);
    }
  }

  @override
  Widget build(BuildContext context) {
    final safePadding = MediaQuery.of(context).padding;

    return Scaffold(
      backgroundColor: AppColors.bg,
      body: Column(
        children: [
          _buildHeader(safePadding.top),
          Expanded(
            child: _conv.messages.isEmpty && !_isStreaming
                ? _buildWelcome()
                : ListView.builder(
              controller: _scrollCtrl,
              padding: EdgeInsets.fromLTRB(
                  16, 12, 16, safePadding.bottom + 8),
              itemCount: _conv.messages.length +
                  (_isStreaming ? 1 : 0) +
                  (_isLoading && !_isStreaming ? 1 : 0),
              itemBuilder: (_, i) {
                if (i < _conv.messages.length) {
                  return _buildMessage(_conv.messages[i]);
                }
                if (_isStreaming) return _buildStreamingBubble();
                return _buildTypingIndicator();
              },
            ),
          ),
          _buildInputBar(safePadding.bottom),
        ],
      ),
    );
  }

  Widget _buildHeader(double topPadding) {
    return Container(
      padding: EdgeInsets.only(
        top: topPadding + 8,
        bottom: 12,
        left: 8,
        right: 16,
      ),
      decoration: const BoxDecoration(
        color: AppColors.primary,
        borderRadius: BorderRadius.only(
          bottomLeft: Radius.circular(24),
          bottomRight: Radius.circular(24),
        ),
      ),
      child: Row(
        children: [
          IconButton(
            icon: const Icon(Icons.arrow_back_ios_rounded,
                color: Colors.white, size: 20),
            onPressed: () => Navigator.pop(context),
          ),
          const Text('🌿', style: TextStyle(fontSize: 22)),
          const SizedBox(width: 8),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  _conv.title,
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 15,
                    fontWeight: FontWeight.w700,
                  ),
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                ),
                Text(
                  '${_conv.messages.length} tin nhắn',
                  style: TextStyle(
                    color: Colors.white.withOpacity(0.6),
                    fontSize: 11,
                  ),
                ),
              ],
            ),
          ),
          IconButton(
            icon: const Icon(Icons.delete_outline_rounded,
                color: Colors.white70, size: 22),
            onPressed: _deleteConversation,
          ),
        ],
      ),
    );
  }

  Widget _buildWelcome() {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(32),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Text('🌱', style: TextStyle(fontSize: 72)),
            const SizedBox(height: 16),
            const Text(
              'Bắt đầu chẩn đoán',
              style: TextStyle(
                color: AppColors.textDark,
                fontSize: 20,
                fontWeight: FontWeight.w800,
              ),
            ),
            const SizedBox(height: 8),
            const Text(
              'Chụp hoặc tải ảnh lá cây để AI phân tích bệnh,\nhoặc hỏi bất kỳ điều gì về cây trồng.',
              textAlign: TextAlign.center,
              style: TextStyle(
                color: AppColors.textLight,
                fontSize: 14,
                height: 1.5,
              ),
            ),
            const SizedBox(height: 24),
            Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                _quickTip('📷', 'Chụp ảnh'),
                const SizedBox(width: 12),
                _quickTip('🖼️', 'Tải ảnh lên'),
                const SizedBox(width: 12),
                _quickTip('💬', 'Hỏi về cây'),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _quickTip(String emoji, String label) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: AppColors.bubble,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: AppColors.divider),
      ),
      child: Column(
        children: [
          Text(emoji, style: const TextStyle(fontSize: 20)),
          const SizedBox(height: 4),
          Text(label,
              style: const TextStyle(
                color: AppColors.textMid,
                fontSize: 11,
                fontWeight: FontWeight.w500,
              )),
        ],
      ),
    );
  }

  Widget _buildMessage(ChatMessage msg) {
    switch (msg.type) {
      case MessageType.image:
        return _buildImageBubble(msg);
      case MessageType.text:
        return _buildUserBubble(msg);
      case MessageType.ai:
        return _buildAIBubble(msg.content);
    }
  }

  Widget _buildImageBubble(ChatMessage msg) {
    return Align(
      alignment: Alignment.centerRight,
      child: Container(
        margin: const EdgeInsets.only(bottom: 12, left: 60),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.end,
          children: [
            ClipRRect(
              borderRadius: BorderRadius.circular(16),
              child: msg.imagePath != null && File(msg.imagePath!).existsSync()
                  ? Image.file(
                File(msg.imagePath!),
                height: 200,
                width: double.infinity,
                fit: BoxFit.cover,
              )
                  : Container(
                height: 100,
                color: AppColors.surface,
                child: const Center(
                  child: Text('🖼️ Ảnh không còn tồn tại',
                      style: TextStyle(color: AppColors.textLight)),
                ),
              ),
            ),
            // Hiển thị custom prompt nếu người dùng có nhập
            if (msg.content.isNotEmpty && msg.content != 'Ảnh lá cây')
              Container(
                margin: const EdgeInsets.only(top: 6),
                padding:
                const EdgeInsets.symmetric(horizontal: 12, vertical: 7),
                decoration: const BoxDecoration(
                  color: AppColors.primary,
                  borderRadius: BorderRadius.only(
                    topLeft: Radius.circular(14),
                    topRight: Radius.circular(14),
                    bottomLeft: Radius.circular(14),
                    bottomRight: Radius.circular(4),
                  ),
                ),
                child: Text(
                  msg.content,
                  style: const TextStyle(
                      color: Colors.white, fontSize: 13, height: 1.4),
                ),
              ),
            const SizedBox(height: 4),
            Text(
              _formatTime(msg.time),
              style: const TextStyle(color: AppColors.textLight, fontSize: 10),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildUserBubble(ChatMessage msg) {
    return Align(
      alignment: Alignment.centerRight,
      child: Container(
        margin: const EdgeInsets.only(bottom: 12, left: 60),
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
        decoration: const BoxDecoration(
          color: AppColors.primary,
          borderRadius: BorderRadius.only(
            topLeft: Radius.circular(18),
            topRight: Radius.circular(18),
            bottomLeft: Radius.circular(18),
            bottomRight: Radius.circular(4),
          ),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.end,
          children: [
            Text(
              msg.content,
              style: const TextStyle(
                  color: Colors.white, fontSize: 14, height: 1.4),
            ),
            const SizedBox(height: 4),
            Text(
              _formatTime(msg.time),
              style: TextStyle(
                  color: Colors.white.withOpacity(0.6), fontSize: 10),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildAIBubble(String content) {
    return Align(
      alignment: Alignment.centerLeft,
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            width: 32,
            height: 32,
            margin: const EdgeInsets.only(right: 8, top: 4),
            decoration: BoxDecoration(
              color: AppColors.accent.withOpacity(0.2),
              shape: BoxShape.circle,
            ),
            child: const Center(
              child: Text('🌿', style: TextStyle(fontSize: 16)),
            ),
          ),
          Flexible(
            child: Container(
              margin: const EdgeInsets.only(bottom: 12, right: 40),
              padding:
              const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
              decoration: BoxDecoration(
                color: AppColors.bubbleAI,
                borderRadius: const BorderRadius.only(
                  topLeft: Radius.circular(4),
                  topRight: Radius.circular(18),
                  bottomLeft: Radius.circular(18),
                  bottomRight: Radius.circular(18),
                ),
                border: Border.all(color: AppColors.divider, width: 1),
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withOpacity(0.05),
                    blurRadius: 4,
                    offset: const Offset(0, 2),
                  ),
                ],
              ),
              child: _buildFormattedText(content),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildStreamingBubble() {
    return Align(
      alignment: Alignment.centerLeft,
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            width: 32,
            height: 32,
            margin: const EdgeInsets.only(right: 8, top: 4),
            decoration: BoxDecoration(
              color: AppColors.accent.withOpacity(0.2),
              shape: BoxShape.circle,
            ),
            child: const Center(
              child: Text('🌿', style: TextStyle(fontSize: 16)),
            ),
          ),
          Flexible(
            child: Container(
              margin: const EdgeInsets.only(bottom: 12, right: 40),
              padding:
              const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
              decoration: BoxDecoration(
                color: AppColors.bubbleAI,
                borderRadius: const BorderRadius.only(
                  topLeft: Radius.circular(4),
                  topRight: Radius.circular(18),
                  bottomLeft: Radius.circular(18),
                  bottomRight: Radius.circular(18),
                ),
                border: Border.all(color: AppColors.divider, width: 1),
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.end,
                children: [
                  Flexible(child: _buildFormattedText(_streamingText)),
                  const SizedBox(width: 4),
                  _buildCursor(),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildCursor() {
    return TweenAnimationBuilder<double>(
      tween: Tween(begin: 0, end: 1),
      duration: const Duration(milliseconds: 500),
      builder: (_, v, __) => Opacity(
        opacity: v > 0.5 ? 1.0 : 0.0,
        child: Container(
          width: 2,
          height: 16,
          color: AppColors.primaryLight,
        ),
      ),
      onEnd: () => setState(() {}),
    );
  }

  Widget _buildTypingIndicator() {
    return Align(
      alignment: Alignment.centerLeft,
      child: Row(
        children: [
          Container(
            width: 32,
            height: 32,
            margin: const EdgeInsets.only(right: 8),
            decoration: BoxDecoration(
              color: AppColors.accent.withOpacity(0.2),
              shape: BoxShape.circle,
            ),
            child: const Center(
              child: Text('🌿', style: TextStyle(fontSize: 16)),
            ),
          ),
          Container(
            margin: const EdgeInsets.only(bottom: 12),
            padding:
            const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
            decoration: BoxDecoration(
              color: AppColors.bubbleAI,
              borderRadius: BorderRadius.circular(18),
              border: Border.all(color: AppColors.divider),
            ),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: List.generate(
                3,
                    (i) => _BounceDot(delay: Duration(milliseconds: i * 150)),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildFormattedText(String text) {
    final spans = <TextSpan>[];
    final regex = RegExp(r'\*\*(.+?)\*\*');
    int last = 0;

    for (final match in regex.allMatches(text)) {
      if (match.start > last) {
        spans.add(TextSpan(
          text: text.substring(last, match.start),
          style: const TextStyle(
              color: AppColors.textDark, fontSize: 14, height: 1.5),
        ));
      }
      spans.add(TextSpan(
        text: match.group(1),
        style: const TextStyle(
          color: AppColors.primary,
          fontSize: 14,
          fontWeight: FontWeight.w700,
          height: 1.5,
        ),
      ));
      last = match.end;
    }

    if (last < text.length) {
      spans.add(TextSpan(
        text: text.substring(last),
        style: const TextStyle(
            color: AppColors.textDark, fontSize: 14, height: 1.5),
      ));
    }

    return RichText(text: TextSpan(children: spans));
  }

  Widget _buildInputBar(double bottomPadding) {
    return Container(
      padding: EdgeInsets.only(
        left: 12,
        right: 12,
        top: 8,
        bottom: bottomPadding + 8,
      ),
      decoration: BoxDecoration(
        color: AppColors.card,
        border: const Border(
          top: BorderSide(color: AppColors.divider, width: 1),
        ),
        boxShadow: [
          BoxShadow(
            color: AppColors.primary.withOpacity(0.08),
            blurRadius: 12,
            offset: const Offset(0, -4),
          ),
        ],
      ),
      child: Row(
        children: [
          // ---- NÚT ẢNH — nhấn để mở menu camera/gallery ----
          GestureDetector(
            onTap: _isLoading ? null : _showImageSourceMenu,
            child: Container(
              width: 42,
              height: 42,
              decoration: BoxDecoration(
                color: _isLoading
                    ? AppColors.surface
                    : AppColors.accent.withOpacity(0.15),
                borderRadius: BorderRadius.circular(12),
              ),
              child: Center(
                child: Text(
                  '📷',
                  style: TextStyle(
                    fontSize: 20,
                    color: _isLoading ? Colors.grey : null,
                  ),
                ),
              ),
            ),
          ),
          const SizedBox(width: 8),

          // ---- TEXT FIELD ----
          Expanded(
            child: Container(
              constraints: const BoxConstraints(maxHeight: 120),
              decoration: BoxDecoration(
                color: AppColors.surface,
                borderRadius: BorderRadius.circular(20),
                border: Border.all(color: AppColors.divider),
              ),
              child: TextField(
                controller: _ctrl,
                maxLines: null,
                enabled: !_isLoading,
                textInputAction: TextInputAction.send,
                onSubmitted: (_) => _sendMessage(),
                style: const TextStyle(
                  color: AppColors.textDark,
                  fontSize: 14,
                ),
                decoration: const InputDecoration(
                  hintText: 'Hỏi về cây trồng...',
                  hintStyle:
                  TextStyle(color: AppColors.textLight, fontSize: 14),
                  contentPadding:
                  EdgeInsets.symmetric(horizontal: 14, vertical: 10),
                  border: InputBorder.none,
                ),
              ),
            ),
          ),
          const SizedBox(width: 8),

          // ---- NÚT GỬI ----
          GestureDetector(
            onTap: _isLoading ? null : _sendMessage,
            child: Container(
              width: 42,
              height: 42,
              decoration: BoxDecoration(
                color: _isLoading ? AppColors.surface : AppColors.primary,
                borderRadius: BorderRadius.circular(12),
              ),
              child: _isLoading
                  ? const Center(
                child: SizedBox(
                  width: 18,
                  height: 18,
                  child: CircularProgressIndicator(
                    strokeWidth: 2,
                    color: AppColors.primaryLight,
                  ),
                ),
              )
                  : const Icon(Icons.send_rounded,
                  color: Colors.white, size: 20),
            ),
          ),
        ],
      ),
    );
  }

  String _formatTime(DateTime dt) {
    final h = dt.hour.toString().padLeft(2, '0');
    final m = dt.minute.toString().padLeft(2, '0');
    return '$h:$m';
  }
}

// ============================================================
// BOTTOM SHEET — CHỌN NGUỒN ẢNH (Camera / Gallery)
// ============================================================

class _ImageSourceSheet extends StatelessWidget {
  final VoidCallback onCamera;
  final VoidCallback onGallery;

  const _ImageSourceSheet({required this.onCamera, required this.onGallery});

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppColors.card,
        borderRadius: BorderRadius.circular(24),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          const SizedBox(height: 12),
          Container(
            width: 36,
            height: 4,
            decoration: BoxDecoration(
              color: AppColors.divider,
              borderRadius: BorderRadius.circular(2),
            ),
          ),
          const SizedBox(height: 20),
          const Text(
            'Chọn nguồn ảnh',
            style: TextStyle(
              color: AppColors.textDark,
              fontSize: 16,
              fontWeight: FontWeight.w700,
            ),
          ),
          const SizedBox(height: 20),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceEvenly,
            children: [
              _SourceOption(
                emoji: '📷',
                label: 'Chụp ảnh',
                subtitle: 'Dùng camera',
                color: AppColors.bubble,
                onTap: onCamera,
              ),
              _SourceOption(
                emoji: '🖼️',
                label: 'Thư viện',
                subtitle: 'Chọn từ máy',
                color: AppColors.surface,
                onTap: onGallery,
              ),
            ],
          ),
          const SizedBox(height: 24),
        ],
      ),
    );
  }
}

class _SourceOption extends StatelessWidget {
  final String emoji;
  final String label;
  final String subtitle;
  final Color color;
  final VoidCallback onTap;

  const _SourceOption({
    required this.emoji,
    required this.label,
    required this.subtitle,
    required this.color,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        width: 140,
        padding: const EdgeInsets.symmetric(vertical: 20),
        decoration: BoxDecoration(
          color: color,
          borderRadius: BorderRadius.circular(18),
          border: Border.all(color: AppColors.divider),
        ),
        child: Column(
          children: [
            Text(emoji, style: const TextStyle(fontSize: 36)),
            const SizedBox(height: 10),
            Text(label,
                style: const TextStyle(
                  color: AppColors.textDark,
                  fontSize: 14,
                  fontWeight: FontWeight.w700,
                )),
            const SizedBox(height: 3),
            Text(subtitle,
                style: const TextStyle(
                    color: AppColors.textLight, fontSize: 11)),
          ],
        ),
      ),
    );
  }
}

// ============================================================
// BOTTOM SHEET — XEM TRƯỚC ẢNH & NHẬP PROMPT
// ============================================================

class _ImagePromptSheet extends StatelessWidget {
  final String imagePath;
  final TextEditingController promptCtrl;
  final VoidCallback onSend;
  final VoidCallback onCancel;

  const _ImagePromptSheet({
    required this.imagePath,
    required this.promptCtrl,
    required this.onSend,
    required this.onCancel,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.fromLTRB(16, 0, 16, 16),
      decoration: BoxDecoration(
        color: AppColors.card,
        borderRadius: BorderRadius.circular(24),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          const SizedBox(height: 12),
          // Drag handle
          Container(
            width: 36,
            height: 4,
            decoration: BoxDecoration(
              color: AppColors.divider,
              borderRadius: BorderRadius.circular(2),
            ),
          ),
          const SizedBox(height: 16),

          // Tiêu đề
          const Padding(
            padding: EdgeInsets.symmetric(horizontal: 20),
            child: Row(
              children: [
                Text('🌿', style: TextStyle(fontSize: 20)),
                SizedBox(width: 8),
                Text(
                  'Xem trước & thêm ghi chú',
                  style: TextStyle(
                    color: AppColors.textDark,
                    fontSize: 15,
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 14),

          // Preview ảnh
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 20),
            child: ClipRRect(
              borderRadius: BorderRadius.circular(16),
              child: File(imagePath).existsSync()
                  ? Image.file(
                File(imagePath),
                height: 160,
                width: double.infinity,
                fit: BoxFit.cover,
              )
                  : Container(
                height: 100,
                color: AppColors.surface,
                child: const Center(
                    child: Text('🖼️',
                        style: TextStyle(fontSize: 40))),
              ),
            ),
          ),
          const SizedBox(height: 14),

          // Ô nhập prompt
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 20),
            child: Container(
              decoration: BoxDecoration(
                color: AppColors.surface,
                borderRadius: BorderRadius.circular(14),
                border: Border.all(color: AppColors.divider),
              ),
              child: TextField(
                controller: promptCtrl,
                maxLines: 3,
                autofocus: false,
                style: const TextStyle(
                    color: AppColors.textDark, fontSize: 14),
                decoration: const InputDecoration(
                  hintText:
                  'Nhập yêu cầu cụ thể... (để trống = dùng mặc định)',
                  hintStyle: TextStyle(
                      color: AppColors.textLight, fontSize: 13),
                  contentPadding: EdgeInsets.all(14),
                  border: InputBorder.none,
                ),
              ),
            ),
          ),

          // Ghi chú nhỏ
          Padding(
            padding:
            const EdgeInsets.symmetric(horizontal: 20, vertical: 8),
            child: Row(
              children: const [
                Icon(Icons.info_outline_rounded,
                    size: 13, color: AppColors.textLight),
                SizedBox(width: 5),
                Flexible(
                  child: Text(
                    'Để trống → AI tự động dùng prompt chẩn đoán mặc định',
                    style: TextStyle(
                        color: AppColors.textLight, fontSize: 11),
                  ),
                ),
              ],
            ),
          ),

          // Nút hành động
          Padding(
            padding: const EdgeInsets.fromLTRB(20, 4, 20, 20),
            child: Row(
              children: [
                Expanded(
                  child: GestureDetector(
                    onTap: onCancel,
                    child: Container(
                      height: 46,
                      decoration: BoxDecoration(
                        color: AppColors.surface,
                        borderRadius: BorderRadius.circular(12),
                        border: Border.all(color: AppColors.divider),
                      ),
                      child: const Center(
                        child: Text(
                          'Huỷ',
                          style: TextStyle(
                            color: AppColors.textMid,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                      ),
                    ),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  flex: 2,
                  child: GestureDetector(
                    onTap: onSend,
                    child: Container(
                      height: 46,
                      decoration: BoxDecoration(
                        color: AppColors.primary,
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: const Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Icon(Icons.send_rounded,
                              color: Colors.white, size: 18),
                          SizedBox(width: 8),
                          Text(
                            'Gửi phân tích',
                            style: TextStyle(
                              color: Colors.white,
                              fontWeight: FontWeight.w700,
                              fontSize: 14,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

// ============================================================
// BOUNCE DOT ANIMATION
// ============================================================

class _BounceDot extends StatefulWidget {
  final Duration delay;
  const _BounceDot({required this.delay});

  @override
  State<_BounceDot> createState() => _BounceDotState();
}

class _BounceDotState extends State<_BounceDot>
    with SingleTickerProviderStateMixin {
  late AnimationController _ctrl;
  late Animation<double> _anim;

  @override
  void initState() {
    super.initState();
    _ctrl = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 600),
    );
    Future.delayed(widget.delay, () {
      if (mounted) _ctrl.repeat(reverse: true);
    });
    _anim = Tween<double>(begin: 0, end: -6).animate(
      CurvedAnimation(parent: _ctrl, curve: Curves.easeInOut),
    );
  }

  @override
  void dispose() {
    _ctrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _anim,
      builder: (_, __) => Transform.translate(
        offset: Offset(0, _anim.value),
        child: Container(
          width: 6,
          height: 6,
          margin: const EdgeInsets.symmetric(horizontal: 3),
          decoration: const BoxDecoration(
            color: AppColors.accent,
            shape: BoxShape.circle,
          ),
        ),
      ),
    );
  }
}