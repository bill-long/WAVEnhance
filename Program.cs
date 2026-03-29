using System.Diagnostics;
using NAudio.Wave;

namespace WAVEnhance;

class Program
{
    static readonly CancellationTokenSource _cts = new();
    static Process? _activeProcess;

    static async Task<int> Main(string[] args)
    {
        Console.CancelKeyPress += (_, e) =>
        {
            e.Cancel = true;
            _cts.Cancel();
            KillActiveProcess();
        };

        var options = ParseArgs(args);
        if (options is null) return 1;

        var wavFiles = DiscoverWavFiles(options.Input);
        if (wavFiles.Count == 0)
        {
            Console.Error.WriteLine($"No WAV files found at: {options.Input}");
            return 1;
        }

        Directory.CreateDirectory(options.Output);

        Console.WriteLine($"WAVEnhance — AI Audio Super-Resolution");
        Console.WriteLine($"  Input:       {options.Input}");
        Console.WriteLine($"  Output:      {options.Output}");
        Console.WriteLine($"  Sample Rate: {options.SampleRate} Hz");
        Console.WriteLine($"  Files:       {wavFiles.Count}");
        Console.WriteLine($"  Guidance:    {options.Guidance}");
        Console.WriteLine($"  Steps:       {options.Steps}");
        Console.WriteLine($"  Seed:        {options.Seed}");
        Console.WriteLine($"  Device:      {options.Device}");
        Console.WriteLine();

        int succeeded = 0, failed = 0;
        var stopwatch = Stopwatch.StartNew();

        for (int i = 0; i < wavFiles.Count; i++)
        {
            if (_cts.IsCancellationRequested)
            {
                Console.WriteLine("\nCancelled by user.");
                break;
            }

            var inputPath = wavFiles[i];
            var fileName = Path.GetFileName(inputPath);
            var outputPath = Path.Combine(options.Output, fileName);

            Console.WriteLine($"[{i + 1}/{wavFiles.Count}] {fileName}");

            // Validate input
            if (!ValidateWavFile(inputPath))
            {
                Console.Error.WriteLine($"  ✗ Skipped (not a valid 16-bit PCM WAV)");
                failed++;
                continue;
            }

            // Run AudioSR via Python
            var enhancedPath = options.SampleRate == 48000
                ? outputPath
                : Path.Combine(options.Output, $"_temp_{fileName}");

            bool success = await RunAudioSR(inputPath, enhancedPath, options);
            if (!success)
            {
                failed++;
                continue;
            }

            // Optionally resample to target sample rate
            if (options.SampleRate != 48000)
            {
                Console.WriteLine($"  Resampling to {options.SampleRate} Hz...");
                Resample(enhancedPath, outputPath, options.SampleRate);
                File.Delete(enhancedPath);
            }

            Console.WriteLine($"  ✓ Done → {outputPath}");
            succeeded++;
        }

        stopwatch.Stop();
        Console.WriteLine();
        Console.WriteLine($"Complete: {succeeded} succeeded, {failed} failed ({stopwatch.Elapsed:hh\\:mm\\:ss})");
        return failed > 0 ? 1 : 0;
    }

    static List<string> DiscoverWavFiles(string path)
    {
        if (File.Exists(path) && path.EndsWith(".wav", StringComparison.OrdinalIgnoreCase))
            return [path];

        if (Directory.Exists(path))
            return Directory.GetFiles(path, "*.wav", SearchOption.TopDirectoryOnly)
                            .OrderBy(f => f)
                            .ToList();

        return [];
    }

    static bool ValidateWavFile(string path)
    {
        try
        {
            using var reader = new WaveFileReader(path);
            var fmt = reader.WaveFormat;
            if (fmt.Encoding != WaveFormatEncoding.Pcm)
            {
                Console.Error.WriteLine($"  Format: {fmt.Encoding} (expected PCM)");
                return false;
            }
            if (fmt.BitsPerSample != 16)
            {
                Console.Error.WriteLine($"  Bit depth: {fmt.BitsPerSample} (expected 16)");
                return false;
            }
            Console.WriteLine($"  {fmt.SampleRate} Hz, {fmt.BitsPerSample}-bit, {fmt.Channels}ch, {reader.TotalTime:mm\\:ss}");
            return true;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"  Error reading WAV: {ex.Message}");
            return false;
        }
    }

    static async Task<bool> RunAudioSR(string inputPath, string outputPath, Options options)
    {
        // Find Python in the .venv
        var venvPython = FindVenvPython();
        if (venvPython is null)
        {
            Console.Error.WriteLine("  ✗ Python venv not found. Run setup first.");
            return false;
        }

        var scriptPath = Path.Combine(AppContext.BaseDirectory, "enhance.py");
        // Fall back to script next to the project file
        if (!File.Exists(scriptPath))
            scriptPath = Path.Combine(Path.GetDirectoryName(Environment.ProcessPath ?? ".")!, "enhance.py");
        if (!File.Exists(scriptPath))
            scriptPath = Path.Combine(Directory.GetCurrentDirectory(), "enhance.py");

        var psi = new ProcessStartInfo
        {
            FileName = venvPython,
            ArgumentList =
            {
                scriptPath,
                "--input", inputPath,
                "--output", outputPath,
                "--guidance", options.Guidance.ToString(),
                "--steps", options.Steps.ToString(),
                "--seed", options.Seed.ToString(),
                "--device", options.Device,
                "--chunk-seconds", options.ChunkSeconds.ToString(),
                "--overlap", options.Overlap.ToString()
            },
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true
        };
        psi.Environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True";

        using var process = Process.Start(psi);
        if (process is null)
        {
            Console.Error.WriteLine("  ✗ Failed to start Python process");
            return false;
        }

        _activeProcess = process;

        // Read stdout for progress
        string? lastError = null;
        bool success = false;

        var stdoutTask = Task.Run(() =>
        {
            while (process.StandardOutput.ReadLine() is { } line)
            {
                if (line.StartsWith("PROGRESS:"))
                {
                    var parts = line.Split(':', 3);
                    if (parts.Length == 3)
                        Console.WriteLine($"  [{parts[1],3}%] {parts[2]}");
                }
                else if (line.StartsWith("RESULT:SUCCESS:"))
                {
                    success = true;
                }
                else if (line.StartsWith("RESULT:ERROR:"))
                {
                    lastError = line["RESULT:ERROR:".Length..];
                }
            }
        });

        var stderrTask = Task.Run(() =>
        {
            while (process.StandardError.ReadLine() is { } line)
            {
                // Ignore Python warnings — only capture actual errors
                if (line.Contains("Warning:", StringComparison.OrdinalIgnoreCase) ||
                    line.Contains("FutureWarning", StringComparison.OrdinalIgnoreCase) ||
                    line.Contains("UserWarning", StringComparison.OrdinalIgnoreCase) ||
                    line.TrimStart().StartsWith("warnings.warn("))
                    continue;

                if (!string.IsNullOrWhiteSpace(line))
                    lastError = line;
            }
        });

        await process.WaitForExitAsync(_cts.Token).ConfigureAwait(ConfigureAwaitOptions.SuppressThrowing);
        await Task.WhenAll(stdoutTask, stderrTask);
        _activeProcess = null;

        if (_cts.IsCancellationRequested)
        {
            KillActiveProcess(process);
            return false;
        }

        if (!success || process.ExitCode != 0)
        {
            Console.Error.WriteLine($"  ✗ AudioSR failed: {lastError ?? "unknown error"}");
            return false;
        }

        return true;
    }

    static void KillActiveProcess(Process? process = null)
    {
        process ??= _activeProcess;
        if (process is null || process.HasExited) return;
        try
        {
            process.Kill(entireProcessTree: true);
            process.WaitForExit(3000);
        }
        catch { /* already exited */ }
    }

    static string? FindVenvPython()
    {

        string[] searchRoots =
        [
            Path.GetDirectoryName(Environment.ProcessPath ?? ".") ?? ".",
            Directory.GetCurrentDirectory(),
        ];

        bool isWindows = OperatingSystem.IsWindows();
        string[] venvDirs = [".venv", "venv"];

        foreach (var root in searchRoots)
        {
            foreach (var venv in venvDirs)
            {
                var candidate = isWindows
                    ? Path.Combine(root, venv, "Scripts", "python.exe")
                    : Path.Combine(root, venv, "bin", "python");

                if (File.Exists(candidate))
                    return candidate;
            }
        }

        return null;
    }

    static void Resample(string inputPath, string outputPath, int targetSampleRate)
    {
        using var reader = new WaveFileReader(inputPath);
        var sourceFormat = reader.WaveFormat;

        var outFormat = new WaveFormat(targetSampleRate, sourceFormat.BitsPerSample, sourceFormat.Channels);
        using var resampler = new MediaFoundationResampler(reader, outFormat)
        {
            ResamplerQuality = 60  // highest quality (1-60)
        };

        WaveFileWriter.CreateWaveFile(outputPath, resampler);
    }

    static Options? ParseArgs(string[] args)
    {
        var options = new Options();
        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--input" or "-i":
                    options.Input = args[++i];
                    break;
                case "--output" or "-o":
                    options.Output = args[++i];
                    break;
                case "--sample-rate" or "-s":
                    options.SampleRate = int.Parse(args[++i]);
                    break;
                case "--guidance":
                    options.Guidance = double.Parse(args[++i]);
                    break;
                case "--steps":
                    options.Steps = int.Parse(args[++i]);
                    break;
                case "--seed":
                    options.Seed = int.Parse(args[++i]);
                    break;
                case "--device" or "-d":
                    options.Device = args[++i];
                    break;
                case "--chunk-seconds":
                    options.ChunkSeconds = double.Parse(args[++i]);
                    break;
                case "--overlap":
                    options.Overlap = double.Parse(args[++i]);
                    break;
                case "--help" or "-h":
                    PrintUsage();
                    return null;
                default:
                    Console.Error.WriteLine($"Unknown option: {args[i]}");
                    PrintUsage();
                    return null;
            }
        }

        if (string.IsNullOrEmpty(options.Input) || string.IsNullOrEmpty(options.Output))
        {
            Console.Error.WriteLine("Error: --input and --output are required.");
            PrintUsage();
            return null;
        }

        if (options.SampleRate != 48000 && options.SampleRate != 96000)
        {
            Console.Error.WriteLine("Error: --sample-rate must be 48000 or 96000.");
            return null;
        }

        return options;
    }

    static void PrintUsage()
    {
        Console.WriteLine("""
        
        Usage: WAVEnhance --input <path> --output <path> [options]

        AI-powered audio super-resolution using AudioSR.
        Restores lost high-frequency content in WAV files that were
        transcoded from lossy sources (MP3/AAC).

        Options:
          --input, -i        Input WAV file or directory (required)
          --output, -o       Output directory (required)
          --sample-rate, -s  Target sample rate: 48000 or 96000 (default: 48000)
          --guidance         AudioSR guidance scale (default: 3.5)
          --steps            AudioSR DDIM steps (default: 50)
          --seed             Random seed for reproducibility (default: 42)
          --device, -d       Compute device: auto, cpu, cuda, directml (default: auto)
          --chunk-seconds    Audio chunk length in seconds (default: 10.24)
          --overlap          Crossfade overlap in seconds between chunks (default: 1.0)
          --help, -h         Show this help message
        """);
    }
}

class Options
{
    public string Input { get; set; } = "";
    public string Output { get; set; } = "";
    public int SampleRate { get; set; } = 48000;
    public double Guidance { get; set; } = 3.5;
    public int Steps { get; set; } = 50;
    public int Seed { get; set; } = 42;
    public string Device { get; set; } = "auto";
    public double ChunkSeconds { get; set; } = 10.24;
    public double Overlap { get; set; } = 1.0;
}
