#include <iostream>
#include <string>
#include <coroutine>
#include <exception>
#include <optional>
#include <utility>

class MyTask
{
public:
    // выяснить почему промис тайп создается раньше майтаск
    struct promise_type
    {
        std::optional<std::string> m_resultValue;
        std::exception_ptr m_exception;

        MyTask get_return_object()
        {
            return MyTask{std::coroutine_handle<promise_type>::from_promise(*this)};
        }

        std::suspend_never initial_suspend() noexcept
        {
            return {};
        }

        // почему не suspend_never
        std::suspend_always final_suspend() noexcept
        {
            return {};
        }

        void return_value(std::string value)
        {
            m_resultValue = std::move(value);
        }

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
        if (m_handle)
        {
            m_handle.destroy();
        }
    }

    MyTask(MyTask &&other) noexcept
            : m_handle(std::exchange(other.m_handle, nullptr))
    {}

    MyTask &operator=(MyTask &&other) noexcept
    {
        if (this != &other)
        {
            if (m_handle)
            {
                m_handle.destroy();
            }
            m_handle = std::exchange(other.m_handle, nullptr);
        }
        return *this;
    }

    MyTask(const MyTask &) = delete;

    MyTask &operator=(const MyTask &) = delete;

    std::string GetResult()
    {
        if (!m_handle)
        {
            throw std::logic_error("GetResult called on moved-from task.");
        }

        auto &promise = m_handle.promise();

        if (promise.m_exception)
        {
            std::rethrow_exception(promise.m_exception);
        }

        if (promise.m_resultValue.has_value())
        {
            return *promise.m_resultValue;
        } else
        {
            throw std::logic_error("Coroutine did not produce a result.");
        }
    }
};

// где хранится рзельтат возвращаемый co_return
MyTask SimpleCoroutine()
{
    co_return "Hello from coroutine!";
}

int main()
{
    MyTask task = SimpleCoroutine();
    std::cout << task.GetResult() << std::endl;
    return 0;
}
