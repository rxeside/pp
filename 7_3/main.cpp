#include <iostream>
#include <coroutine>
#include <utility>

struct MyAwaiter
{
    int a, b;
    int result = 0;

    MyAwaiter(int x, int y) : a(x), b(y)
    {}

    // что происходит после await_ready() false
    static bool await_ready() noexcept
    {
        return false;
    }

    // разобраться с noexcept у методов await
    void await_suspend(std::coroutine_handle<> h)
    {
        result = a + b;
    }

    [[nodiscard]] int await_resume() const noexcept
    {
        return result;
    }
};

class MyTask
{
public:
    struct promise_type
    {
        std::exception_ptr m_exception;

        MyTask get_return_object()
        {
            return MyTask{
                    std::coroutine_handle<promise_type>::from_promise(*this)
            };
        }

        static std::suspend_never initial_suspend() noexcept
        {
            return {};
        }

        static std::suspend_always final_suspend() noexcept
        {
            return {};
        }

        void return_void() noexcept
        {}

        void unhandled_exception()
        {
            m_exception = std::current_exception();
        }
    };

private:
    std::coroutine_handle<promise_type> m_handle;

public:
    explicit MyTask(std::coroutine_handle<promise_type> handle)
            : m_handle(handle)
    {}

    ~MyTask()
    {
        if (m_handle) m_handle.destroy();
    }

    MyTask(MyTask &&other) noexcept
            : m_handle(std::exchange(other.m_handle, nullptr))
    {}

    MyTask &operator=(MyTask &&other) noexcept
    {
        if (this != &other)
        {
            if (m_handle) m_handle.destroy();
            m_handle = std::exchange(other.m_handle, nullptr);
        }
        return *this;
    }

    MyTask(const MyTask &) = delete;

    MyTask &operator=(const MyTask &) = delete;

    void Resume()
    {
        if (m_handle && !m_handle.done())
        {
            m_handle.resume();
        }
    }
};

// что нужно сделать с майтаск чтобы его тоже можно было использовать co_await
MyTask CoroutineWithAwait(int x, int y)
{
    std::cout << "Before await\n";
    int result = co_await MyAwaiter(x, y);
    std::cout << result << "\n";
    std::cout << "After await\n";
}

int main()
{
    auto task = CoroutineWithAwait(30, 12);
    std::cout << "Before resume\n";
    task.Resume();
    std::cout << "After resume\n";

    CoroutineWithAwait(5, 10).Resume();

    std::cout << "End of main\n";
}
