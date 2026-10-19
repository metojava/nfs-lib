package data;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Convenience entry point for downloading Fashion-MNIST into
 * {@code data/fashion-mnist}. Delegates to
 * {@link MNISTDownloader} with the Zalando Research mirror URL.
 */
public class FashionMNISTDownloader {

    /** Where the four IDX files live on GitHub. */
    private static final String BASE_URL =
        "https://raw.githubusercontent.com/zalandoresearch/"
        + "fashion-mnist/master/data/fashion/";

    private FashionMNISTDownloader() {
    }

    /**
     * Downloads the Fashion-MNIST IDX files if missing.
     *
     * @return the directory holding the files
     * @throws IOException on network or filesystem failure
     */
    public static Path ensureDownloaded() throws IOException {
        return ensureDownloaded(
            Paths.get("data", "fashion-mnist"));
    }

    /**
     * Downloads the Fashion-MNIST IDX files if missing into a
     * custom directory.
     *
     * @param directory target directory
     * @return the same directory
     * @throws IOException on network or filesystem failure
     */
    public static Path ensureDownloaded(Path directory)
            throws IOException {
        new MNISTDownloader(BASE_URL).download(directory);
        System.out.println(
            "Fashion-MNIST files are ready in: "
            + directory.toAbsolutePath());
        return directory;
    }
}
